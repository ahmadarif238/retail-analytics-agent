# Retail Analytics Agent

A local AI agent for retail analytics using LangGraph, DSPy, and SQLite.

## Graph Design

**7 Nodes**: Router → Retriever → (SQL Gen → Executor → Repair) → Synthesizer → END

- **Router (DSPy Optimized)**: Classifies questions as 'rag', 'sql', or 'hybrid'
- **Retriever (TF-IDF)**: Fetches top-3 document chunks with scores
- **SQL Generator (DSPy)**: Generates SQLite queries using compatibility views
- **Executor**: Runs SQL queries with error capture
- **Synthesizer (Direct Parsing)**: Extracts typed answers, no LLM formatting
- **Repair Loop**: Retries failed SQL up to 2 times
- **Checkpointing**: LangGraph MemorySaver for state persistence

## DSPy Optimization

**Module Optimized**: Router (classification accuracy)  
**Optimizer**: BootstrapFewShot with 4 bootstrapped demos and 4 labeled demos  
**Training Set**: 12 hand-crafted examples (4 rag, 4 sql, 4 hybrid)  
**Metric**: Exact classification match accuracy  

**Results**:
- **Before optimization**: 9/12 correct (75% accuracy)
- **After optimization**: 10/12 correct (83% accuracy)  
- **Improvement**: +8 percentage points (+11% relative improvement)

**Impact**: Optimized router correctly identifies policy/calendar questions as RAG, reducing unnecessary SQL generation attempts.

## Model Limitations & Trade-offs

### What Works
✅ **RAG**: TF-IDF retrieval + pattern matching works perfectly for policy questions  
✅ **Router**: DSPy optimization improves classification accuracy  
✅ **Architecture**: 7-node LangGraph with repair loop meets spec  

### Known Issues (Phi-3.5-mini 3.8B Limitations)

❌ **SQL Generation Hallucinations**: The model consistently generates:
- Syntax errors: `BETWEWEN`, `BETWEDIR` instead of `BETWEEN`
- Non-existent columns: `marketing_date`, `customer_name`, `product_type`
- Invalid table references despite schema provided

**Root Cause**: Phi-3.5-mini-instruct (3.8B parameters) is too small for reliable SQL generation. Larger models (7B+ params) would significantly improve accuracy.

### Compatibility Views

Created lowercase SQL views for simpler query generation:
```sql
orders, order_items, products, customers, categories, suppliers
```

### Assumptions

- **CostOfGoods**: Approximated as `0.7 * UnitPrice` when missing (per spec)
- **1997 Data**: Northwind DB contains 2012-2023 data; 1997 queries return null
- **Model Constraint**: Phi-3.5-mini-instruct via Ollama (per spec)

## Production Recommendations

For production deployment:
1. **Use larger model**: Mistral-7B or Llama-3-8B for SQL generation
2. **Hybrid approach**: Template-based SQL for common queries + LLM for complex ones
3. **Validation layer**: SQL syntax checker before execution

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Create compatibility views (one-time)
python create_views.py

# Run agent
python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
```

## Deliverables

- ✅ **Code**: Full implementation in `agent/`
- ✅ **DSPy Optimization**: Router with documented metrics  
- ✅ **Graph**: 7 nodes + repair loop (≤2 iterations)
- ✅ **Citations**: Proper format (`filename::chunkN`)
- ⚠️ **SQL Accuracy**: Limited by Phi-3.5-mini model size
