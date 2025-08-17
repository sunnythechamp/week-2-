import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PromptOptimizer:
    def __init__(self):
        self.task_prompts = {
            "summarization": self._summarization_prompt,
            "classification": self._classification_prompt,
            "generation": self._generation_prompt
        }

    def optimize(self, task_type: str, text: str) -> str:
        """Optimize prompt based on task type."""
        task_type = task_type.lower()
        if task_type not in self.task_prompts:
            logging.error(f"Unsupported task type: {task_type}")
            raise ValueError(f"Task type '{task_type}' not supported.")
        
        logging.info(f"Optimizing prompt for task: {task_type}")
        return self.task_prompts[task_type](text)

    def _summarization_prompt(self, text: str) -> str:
        return f"Summarize the following text in a concise way:\n\n{text}"

    def _classification_prompt(self, text: str) -> str:
        return f"Classify the sentiment (positive, neutral, negative) of the following text:\n\n{text}"

    def _generation_prompt(self, text: str) -> str:
        return f"Write a creative continuation of the following text:\n\n{text}"

# Example Usage
if __name__ == "__main__":
    optimizer = PromptOptimizer()

    sample_text = "Artificial Intelligence is transforming industries by automating tasks and improving decision-making."

    tasks = ["summarization", "classification", "generation"]

    for task in tasks:
        optimized_prompt = optimizer.optimize(task, sample_text)
        print(f"\n--- {task.upper()} PROMPT ---")
        print(optimized_prompt)
