## 默认的对于所有模型均适用的提示词模板翻译如下：

**提示词**
请尽可能有帮助且准确地回答用户。

{{instruction}}

你可以使用以下工具：

{{tools}}

使用一个 JSON 对象来指定要调用的工具，其中包含 {{TOOL\_NAME\_KEY}} 键（工具名称）和 {{ACTION\_INPUT\_KEY}} 键（工具输入）。
合法的 "{{TOOL\_NAME\_KEY}}" 值为："Final Answer" 或 {{tool\_names}}

每个 \$JSON\_BLOB 中只能包含 **一个** action，格式如下：

```
{
"{{TOOL_NAME_KEY}}": $TOOL_NAME,
"{{ACTION_INPUT_KEY}}": $ACTION_INPUT
}
```

请遵循以下格式：
Question: 输入的问题
Thought: 考虑前后步骤
Action:

```
$JSON_BLOB
```

Observation: 动作结果
...（重复 Thought/Action/Observation N 次）
Thought: 我知道应该如何回答
Action:

```
{
"{{TOOL_NAME_KEY}}": "Final Answer",
"{{ACTION_INPUT_KEY}}": "最终给用户的回答"
}
```

开始！请务必记住：**始终返回一个合法的 JSON 对象，且每次只包含一个 action**。如有需要请使用工具，如果合适则可直接回答。格式为：
Action:`$JSON_BLOB`
然后 Observation:.



## 模板的运行逻辑说明

1. Question（问题）：记录用户的原始输入。
2. Thought（思考）：模型需要展示它的推理过程，说明当前要采取什么行动。
3. Action（动作）：当模型决定调用工具时，必须用一个严格符合 JSON 格式的对象来表示调用请求，其中包含：
    a、{{TOOL_NAME_KEY}} ：所使用的工具名称（比如“Final Answer”或其他工具名）；
    b、{{ACTION_INPUT_KEY}} ：传递给工具的输入内容。
4. Observation（观察结果）：展示工具返回的结果。
5. 循环执行：如果一个工具调用不足以得到最终答案，则模型会继续执行 Thought → Action → Observation 的循环，直到获取完整信息。
6. Final Answer（最终回答）：最后，模型必须返回一个包含 "Final Answer" 的 JSON 对象，将最终结果传递给用户。

## 总结
这种模板的最大特点是：
    1、格式统一：不论调用什么工具，调用格式始终是 JSON，方便解析。
    2、过程透明：Thought/Action/Observation 的交替让推理过程可追踪，便于调试和监控。
    3、强制约束：要求模型“每次只输出一个 action”，避免模型在一个回答里乱用多个工具，保证流程稳定。
    
   因此，这个提示词模板不仅是一个“写法规范”，更是 保证 Agent 工具调用稳定性和可控性 的核心机制。在实际开发中，所有基于 ReAct 或工具调用的 Agent 都可以复用这一结构，只需要替换其中的工具列表和业务逻辑即可。