# Migrating to Red Candle 2.0

Red Candle 2.0 unifies the chat API around a single return type, `ChatResponse`,
and replaces the raw token stream with typed events. This is a breaking change
for the chat/tool-calling and streaming surfaces. Embeddings, rerankers, NER,
tokenizers, and `generate`/`generate_stream` are unchanged.

## TL;DR

| 1.x | 2.0 |
|-----|-----|
| `llm.chat(messages)` → `String` | `llm.chat(messages)` → `ChatResponse` (`.content` is the text) |
| `llm.chat_with_tools(messages, tools:)` | `llm.chat(messages, tools:)` |
| `ToolCallResult` | `ChatResponse` |
| `result.text_response` | `response.content` |
| `result.has_tool_calls?` | `response.tool_calls?` |
| `result.tool_results` (Array of Hashes) | `response.tool_calls` (Array of `ToolCall` with `.result`/`.error`) |
| `chat_stream { \|token\| ... }` (String tokens) | `chat_stream { \|event\| ... }` (typed `StreamEvent`) |

## 1. `chat` now returns a `ChatResponse`

In 1.x, `chat` returned a `String`. In 2.0 it returns a `ChatResponse` that always
has the same shape regardless of what the model produced:

```ruby
response = llm.chat(messages)
response.content      # the answer text (thinking + tool markup removed)
response.thinking     # extracted reasoning, or nil
response.tool_calls   # Array<ToolCall>, empty when none
response.raw_response # the unmodified model output
```

`ChatResponse#to_s` (and `#to_str`) return `content`, so the most common 1.x
pattern keeps working unchanged:

```ruby
puts llm.chat(messages)          # still prints the answer
greeting = "Reply: " + llm.chat(messages)  # string coercion still works
```

If you assigned the result and used it as a string, switch to `.content`:

```ruby
# Before
text = llm.chat(messages)
text.upcase

# After
text = llm.chat(messages).content
text.upcase
```

## 2. `chat_with_tools` is removed — use `chat(tools:)`

```ruby
# Before
result = llm.chat_with_tools(messages, tools: [get_weather])
if result.has_tool_calls?
  result.tool_calls.each { |tc| ... }
else
  puts result.text_response
end

# After
response = llm.chat(messages, tools: [get_weather])
if response.tool_calls?
  response.tool_calls.each { |tc| ... }
else
  puts response.content
end
```

### `execute: true`

`ToolCallResult#tool_results` (an array of `{ tool_call:, result:, error: }` hashes)
is gone. With `execute: true`, the result/error is attached directly to each
`ToolCall`:

```ruby
# Before
result = llm.chat_with_tools(messages, tools: tools, execute: true)
result.tool_results.each do |tr|
  puts "#{tr[:tool_call].name} => #{tr[:error] || tr[:result]}"
end

# After
response = llm.chat(messages, tools: tools, execute: true)
response.tool_calls.each do |tc|
  puts "#{tc.name} => #{tc.error || tc.result}"   # tc.success? / tc.executed? also available
end
```

## 3. `ToolCallResult` is removed

It is fully replaced by `ChatResponse`. Update any `is_a?(Candle::ToolCallResult)`
checks to `is_a?(Candle::ChatResponse)`.

## 4. Streaming yields typed events

`chat_stream` no longer yields `String` tokens. It yields `Candle::StreamEvent`
objects with a `type` (`:thinking`, `:content`, `:tool_call`, `:done`) and a
`delta` (the text, for content/thinking):

```ruby
# Before
llm.chat_stream(messages) { |token| print token }

# After
llm.chat_stream(messages) do |event|
  print event.delta if event.content?
end
```

This lets you separate reasoning, answer text, and tool calls as they stream:

```ruby
llm.chat_stream(messages, tools: tools) do |event|
  case event.type
  when :thinking  then log_reasoning(event.delta)
  when :content   then print event.delta
  when :tool_call then handle(event.tool_call)
  when :done      then finalize
  end
end
```

> Note: `generate_stream` is **not** affected — it still yields raw `String`
> tokens. Only the chat-level `chat_stream` changed.

## 5. Thinking extraction is now built in

In 1.x you stripped `<think>` blocks yourself. In 2.0, `chat` extracts reasoning
into `response.thinking` and keeps `response.content` clean. Tags are inferred
from the model (Qwen3, QwQ, DeepSeek-R1 use `<think>...</think>`); other models
default to no extraction.

Override or disable as needed:

```ruby
llm.thinking_parser = ["<thinking>", "</thinking>"]   # per-model override
llm.chat(messages, thinking: false)                   # disable for one call
llm.chat(messages, thinking: ["<reason>", "</reason>"]) # custom tags for one call
Candle::LLM.register_thinking_tags(/my-model/i, "<think>", "</think>")  # global
```

## 6. `Candle::Agent`

The agent's public interface (`Agent#run` → `AgentResult`) is unchanged. It now
uses `chat(tools:, execute:)` internally; if you subclassed or stubbed it against
`chat_with_tools`, update those references.
