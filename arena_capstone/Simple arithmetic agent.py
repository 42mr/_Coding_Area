# Simple arithmetic agent

class ArithmeticTask:

    def __init__(self, num1: int | float, num2: int | float, operations: Optional[list[str]] = None):
        self.num1 = num1
        self.num2 = num2
        self.operations = operations if operations else ["+", "-", "*", "/", "**", "//", "%"]
        self.current_task_number = 0

    def _generate_answers(self) -> list[str]:
        """
        Generates a list of the correct answers for all the possible tasks

        Returns:
            list[str]: A list of the correct answers for all the possible tasks
        """
        answers = []
        for op in self.operations:
            try:
                result = evaluate_expression(f"{self.num1} {op} {self.num2}")
                answers.append(str(result))
            except Exception as e:
                answers.append(f"Error: {str(e)}")
        return answers

    @property
    def get_current_task(self) -> str:
        return f"{self.num1} {self.operations[self.current_task_number]} {self.num2}"
    def update_current_task(self) -> None:
        """
        Increments self.current_task_number by one (modulo the number of operations)
        """
        self.current_task_number = (self.current_task_number + 1) % len(self.operations)
    def get_current_instruction(self) -> ChatMessageUser:
        return ChatMessageUser(content= f"Calculate the following expression {self.get_current_task}. Give your answer in the format <ANSWER>NUMBER</ANSWER> where NUMBER is a numerical value formatted as a float.")

arithmetic_task1 = ArithmeticTask(3, 5)
print(arithmetic_task1.get_current_task)
arithmetic_task1.update_current_task()
print(arithmetic_task1.get_current_task)
print(arithmetic_task1.get_current_instruction())

@tool 
def tool_name():
    def execute(sentence : str, n : int) -> str:
        """
        This tool appends a number to the end of the string.

        Args:
            sentence: this is the string to which you want to append a number.
            n: this is the number you want to append.

        Returns: 
            The sentence with the number appended.
        """
        return sentence + str(n)
    
@tool
def calculate(): 
    async def execute(expression : str) -> str:
        """
        A calculator that can evaluate arithmetic expressions. The input is a mathematical expression, as a string, and the output is the result of the calculation.

        Args:
            expression: the arithmetic expression to evaluate.

        Returns: 
            The result of the calculation, as a string. Or error if the expression is invalid.
        """
        try: 
            return str(evaluate_expression(expression))
        except Exception as e:
            return f"Error: {e}"
    return execute

@agent
def arithmetic_agent(task : ArithmeticTask):

    async def execute(state: AgentState) -> AgentState:
        answer_list = ["wrong"] * len(task.operations)
        success = False
        while not success:
            state.messages.append(task.get_current_instruction())
            state.output = await get_model().generate(input = state.messages, tools = [calculate()], tool_choice = "auto")
            state.messages.append(state.output.message)
            if state.output.message.tool_calls:
                messages, state.output = await execute_tools(state.messages, tools = [calculate()])
                state.messages.extend(messages)
            state.output = await get_model().generate(input = state.messages, tools = [calculate()], tool_choice = "none")
            state.messages.append(state.output.message)
            try:
                if extract_answer(state.output.message.content) == task._generate_answers()[task.current_task_number]:
                    answer_list[task.current_task_number] = extract_answer(state.output.message.content)
                    task.update_current_task()

                else:
                    state.messages.append(ChatMessageUser(content="Incorrect answer. Try again."))
            except IndexError:
                state.messages.append(ChatMessageUser(content="Error: Could not extract answer"))
            if all(ans == task._generate_answers()[i] for i, ans in enumerate(answer_list)):
                success = True
        return state 
    return execute

@task
def agent_task() -> str:
    return Task(dataset = [Sample(input = "", target = "")], message_limit=40)

eval(agent_task(), solver = as_solver(arithmetic_agent(task = ArithmeticTask(3, 5))))