class Parser:
    def parse(self, response):
        if isinstance(response, dict) and 'content' in response:
            response = response['content']
        oring_content = response.replace("\_", "_")
        content = oring_content.replace("\\", "")
        
        try:
            start_pos = content.find("```python")
            if start_pos != -1:
                content = content[start_pos+len("```python"):]

            end_pos = content.find("```")
            if end_pos != -1:
                content = content[:end_pos]
                
            if start_pos == -1:
                return {'status': False, 'content': content, 'message': 'No tool call', 'error_code': 'NOTOOL'}
            
            if start_pos == -1 or end_pos == -1:
                return {'status': False, 'content': content, 'message': 'Program is NOT enclosed in ```python``` properly.', 'error_code': 'unknown'}
            if len(content) > 0:
                compile(content, "prog.py", "exec")
                return {'status': True, 'content': content, 'message': 'Parsing succeeded.', 'error_code': ''}
            else:
                return {'status': False, 'content': content, 'message': "The content is empty, or it failed to parse the content correctly.", 'error_code': 'unknown'}
        except Exception as err:
            return {'status': False, 'content': content, 'message': f"Unexpected {type(err)}: {err}.", 'error_code': 'unknown'}
        
    def trim_to_action_end(self, text):
        first_code_start = text.find("```")
        if first_code_start == -1:
            return text  # 没有代码块，直接返回原文

        first_code_end = text.find("```", first_code_start + 3)
        if first_code_end == -1:
            return text  # 没找到结束符，直接返回原文

        # 返回从开头到第一个代码块结束的位置（包含 ```）
        return text[:first_code_end + 3].rstrip()
        # last_code_block_start = text.rfind("```")
        # if last_code_block_start == -1:
        #     return text  # no code block found
        # # Now find the preceding ``` that starts the block
        # preceding_code_block_start = text.rfind("```", 0, last_code_block_start)
        # if preceding_code_block_start == -1:
        #     return text  # malformed block
        # return text[:last_code_block_start + 3]  # include closing ```
    
def main():
    parser = Parser()
    
    # testing 1
    program = """THOUGHT 0: The task involves moving from the current location (the hall) to the kitchen, then to a bedroom, and finally to another bedroo
m. The task has not started yet, so the initial observation points to the start position.
```python
retrieved_mem = retrieve_memories(memory, query_text="start location of the task")
display(retrieved_mem)
```
OBSERVATION: Execution success with retrieved results.
THOUGHT 1: Based on the retrieved memories, the initial observation is at a hallway. The current step is to proceed to the kitchen. The c
orrect direction to move in this scenario, given the frontier, is forward (front) towards the kitchen.
ANSWER: None
FRONTIER: 1
"""
    program += """def solve():\n"""
    program += """    output0 = text_generation(prompt="Would you rather have an Apple Watch - or a BABY?")\n"""  
    program += """    output1 = text_summarization(text=output0["text"])\n"""
    program += """    return output1\n""" 
    program += """```"""
    program += """HELLO WORLD"""
    #print(program)
    results = parser.parse(program)
    #print(results)

    print(parser.trim_to_action_end(program))
    
if __name__ == '__main__':
    main()