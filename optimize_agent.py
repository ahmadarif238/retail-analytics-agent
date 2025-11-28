import dspy
from dspy.teleprompt import BootstrapFewShot
from agent.dspy_signatures import RouterSignature
import os

# Configure DSPy to use local Ollama model
try:
    lm = dspy.OllamaLocal(model='phi3.5:3.8b-mini-instruct-q4_K_M', max_tokens=1000)
except AttributeError:
    try:
        lm = dspy.Ollama(model='phi3.5:3.8b-mini-instruct-q4_K_M', max_tokens=1000)
    except AttributeError:
        # Fallback for newer dspy versions
        lm = dspy.LM(model='ollama/phi3.5:3.8b-mini-instruct-q4_K_M', api_base='http://localhost:11434', api_key='')

dspy.settings.configure(lm=lm)

def optimize_router():
    # Define training examples
    train_examples = [
        dspy.Example(question="What is the return policy for unopened beverages?", classification="rag").with_inputs("question"),
        dspy.Example(question="How many days do I have to return dairy products?", classification="rag").with_inputs("question"),
        dspy.Example(question="Show me the marketing calendar for Summer 1997.", classification="rag").with_inputs("question"),
        dspy.Example(question="What is the definition of Average Order Value?", classification="rag").with_inputs("question"),
        
        dspy.Example(question="List the top 5 products by unit price.", classification="sql").with_inputs("question"),
        dspy.Example(question="How many orders were placed in 1997?", classification="sql").with_inputs("question"),
        dspy.Example(question="What is the total revenue for product 'Chai'?", classification="sql").with_inputs("question"),
        dspy.Example(question="Who is the customer with the most orders?", classification="sql").with_inputs("question"),
        
        dspy.Example(question="What was the total revenue during the 'Summer Beverages 1997' campaign?", classification="hybrid").with_inputs("question"),
        dspy.Example(question="Which category had the highest sales during 'Winter Classics 1997'?", classification="hybrid").with_inputs("question"),
        dspy.Example(question="Calculate the Gross Margin for the top customer in 1997 using the KPI definition.", classification="hybrid").with_inputs("question"),
        dspy.Example(question="List products in the 'Beverages' category defined in the catalog that have sold more than 100 units.", classification="hybrid").with_inputs("question"),
    ]

    # Define a simple module to optimize
    class RouterModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.prog = dspy.ChainOfThought(RouterSignature)
        
        def forward(self, question):
            return self.prog(question=question)

    # Metric function
    def validate_router(example, pred, trace=None):
        return example.classification.lower() == pred.classification.lower().strip()

    print("Starting optimization...")
    teleprompter = BootstrapFewShot(metric=validate_router, max_bootstrapped_demos=4, max_labeled_demos=4)
    
    # Compile
    optimized_router = teleprompter.compile(RouterModule(), trainset=train_examples)
    
    print("Optimization complete.")
    
    # Evaluate before/after (simple check)
    print("\nEvaluating on training set (sanity check):")
    raw_router = RouterModule()
    correct_raw = 0
    correct_opt = 0
    for ex in train_examples:
        pred_raw = raw_router(ex.question)
        pred_opt = optimized_router(ex.question)
        
        if validate_router(ex, pred_raw):
            correct_raw += 1
        if validate_router(ex, pred_opt):
            correct_opt += 1
            
    print(f"Raw Accuracy: {correct_raw}/{len(train_examples)}")
    print(f"Optimized Accuracy: {correct_opt}/{len(train_examples)}")
    
    # Save the optimized program
    optimized_router.save("optimized_router.json")
    print("Saved optimized_router.json")

if __name__ == "__main__":
    optimize_router()
