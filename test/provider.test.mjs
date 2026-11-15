import assert from "node:assert/strict";
import test from "node:test";

import registerCommandCode from "../index.ts";

let provider;
registerCommandCode({
	registerProvider(name, config) {
		assert.equal(name, "commandcode");
		provider = config;
	},
});

const model = {
	id: "deepseek/deepseek-v4-flash",
	name: "DeepSeek V4 Flash",
	api: "commandcode-generate",
	provider: "commandcode",
	reasoning: true,
	input: ["text"],
	contextWindow: 1_000_000,
	maxTokens: 64_000,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};

test("model registry includes the current OSS roster and vision capabilities", () => {
	const byID = new Map(provider.models.map((entry) => [entry.id, entry]));
	for (const id of [
		"moonshotai/Kimi-K3",
		"zai-org/GLM-5.2-Fast",
		"tencent/Hy3",
		"thinkingmachines/inkling",
	]) {
		assert.ok(byID.has(id), `missing current model ${id}`);
	}
	for (const id of [
		"moonshotai/Kimi-K3",
		"moonshotai/Kimi-K2.7-Code",
		"moonshotai/Kimi-K2.7-Code-Highspeed",
		"moonshotai/Kimi-K2.6",
		"moonshotai/Kimi-K2.5",
		"MiniMaxAI/MiniMax-M3",
		"Qwen/Qwen3.7-Plus",
		"stepfun/Step-3.7-Flash",
		"thinkingmachines/inkling",
	]) {
		assert.ok(byID.get(id)?.input.includes("image"), `${id} should accept image input`);
	}
});

test("truncated NDJSON stream fails instead of emitting done", async (t) => {
	t.mock.method(globalThis, "fetch", async () => {
		return new Response(
			[
				'{"type":"start"}',
				'{"type":"text-start","id":"txt-0"}',
				'{"type":"text-delta","id":"txt-0","text":"partial"}',
			].join("\n"),
			{ status: 200, headers: { "content-type": "application/x-ndjson" } },
		);
	});

	const events = [];
	const stream = provider.streamSimple(model, { systemPrompt: "", messages: [], tools: [] }, { apiKey: "user_test" });
	for await (const event of stream) {
		events.push(event);
	}

	assert.equal(events.at(-1)?.type, "error");
	assert.match(events.at(-1)?.error?.errorMessage ?? "", /terminal event/i);
});

test("tool-call accepts the current args fallback", async (t) => {
	t.mock.method(globalThis, "fetch", async (_input, init) => {
		assert.equal(init.headers["x-command-code-version"], "0.52.1");
		return new Response(
			[
				'{"type":"start"}',
				'{"type":"tool-call","toolCallId":"call_1","toolName":"get_weather","args":{"location":"Paris"}}',
				'{"type":"finish","finishReason":"tool-calls","totalUsage":{"inputTokens":10,"outputTokens":3,"totalTokens":13}}',
			].join("\n"),
			{ status: 200, headers: { "content-type": "application/x-ndjson" } },
		);
	});

	const events = [];
	const stream = provider.streamSimple(model, { systemPrompt: "", messages: [], tools: [] }, { apiKey: "user_test" });
	for await (const event of stream) {
		events.push(event);
	}

	const toolEnd = events.find((event) => event.type === "toolcall_end");
	assert.deepEqual(toolEnd?.toolCall?.arguments, { location: "Paris" });
	assert.equal(events.at(-1)?.message?.usage?.output, 3);
});

test("CMD_ZDR enables the strict zero-retention header", async (t) => {
	const previous = process.env.CMD_ZDR;
	process.env.CMD_ZDR = "1";
	t.after(() => {
		if (previous === undefined) delete process.env.CMD_ZDR;
		else process.env.CMD_ZDR = previous;
	});

	let zdrHeader;
	t.mock.method(globalThis, "fetch", async (_input, init) => {
		zdrHeader = init.headers["x-cmd-zdr"];
		return new Response('{"type":"finish","finishReason":"stop","totalUsage":{"totalTokens":0}}\n', {
			status: 200,
			headers: { "content-type": "application/x-ndjson" },
		});
	});

	const events = [];
	const stream = provider.streamSimple(model, { systemPrompt: "", messages: [], tools: [] }, { apiKey: "user_test" });
	for await (const event of stream) events.push(event);

	assert.equal(zdrHeader, "1");
	assert.equal(events.at(-1)?.type, "done");
});

test("interrupted tool call gets a synthetic result so the request stays valid", async (t) => {
	let sent;
	t.mock.method(globalThis, "fetch", async (_input, init) => {
		sent = JSON.parse(init.body);
		return new Response('{"type":"finish","finishReason":"stop","totalUsage":{"totalTokens":0}}\n', {
			status: 200,
			headers: { "content-type": "application/x-ndjson" },
		});
	});

	// Reproduces a transcript pi leaves behind after Stop/Esc: the assistant
	// message carries a toolCall whose toolResult was never persisted. The
	// gateway rejects such a request with
	// "Tool result is missing for tool call call_00_...".
	const interrupted = {
		role: "assistant",
		content: [
			{ type: "text", text: "Checking the device node." },
			{ type: "toolCall", id: "call_00_kXURtf48dOOoBPfBvdaE0811", name: "bash", arguments: {} },
		],
	};
	const followUp = { role: "user", content: [{ type: "text", text: "continue" }] };
	const trailing = { role: "assistant", content: [{ type: "toolCall", id: "call_orphan_tail", name: "bash", arguments: {} }] };

	const stream = provider.streamSimple(
		model,
		{ systemPrompt: "", messages: [interrupted, followUp, trailing], tools: [] },
		{ apiKey: "user_test" },
	);
	for await (const event of stream) void event;

	const messages = sent.params.messages;
	const callIDs = messages
		.flatMap((m) => (m.role === "assistant" ? m.content : []))
		.filter((b) => b.type === "tool-call")
		.map((b) => b.toolCallId);
	const resultIDs = messages
		.flatMap((m) => (m.role === "tool" ? m.content : []))
		.filter((b) => b.type === "tool-result")
		.map((b) => b.toolCallId);

	// Every tool-call in the transcript must be answered by a tool-result,
	// including the trailing one at the very end of the conversation.
	assert.deepEqual(callIDs, ["call_00_kXURtf48dOOoBPfBvdaE0811", "call_orphan_tail"]);
	assert.deepEqual(resultIDs, callIDs);
	for (const m of messages.filter((m) => m.role === "tool")) {
		for (const b of m.content) assert.match(b.output.value, /interrupted/i);
	}
});

test("answered tool calls are left untouched (no synthetic results)", async (t) => {
	let sent;
	t.mock.method(globalThis, "fetch", async (_input, init) => {
		sent = JSON.parse(init.body);
		return new Response('{"type":"finish","finishReason":"stop","totalUsage":{"totalTokens":0}}\n', {
			status: 200,
			headers: { "content-type": "application/x-ndjson" },
		});
	});

	const messages = [
		{ role: "assistant", content: [{ type: "toolCall", id: "call_1", name: "bash", arguments: { command: "ls" } }] },
		{
			role: "toolResult",
			toolCallId: "call_1",
			toolName: "bash",
			content: [{ type: "text", text: "file.txt" }],
			isError: false,
		},
		{ role: "user", content: [{ type: "text", text: "thanks" }] },
	];

	const stream = provider.streamSimple(model, { systemPrompt: "", messages, tools: [] }, { apiKey: "user_test" });
	for await (const event of stream) void event;

	const toolMsgs = sent.params.messages.filter((m) => m.role === "tool");
	assert.equal(toolMsgs.length, 1);
	assert.equal(toolMsgs[0].content.length, 1);
	assert.equal(toolMsgs[0].content[0].output.value, "file.txt");
});
