from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def ask_question(question, top_k=3, max_length=512):
    # Step 1: Retrieve relevant docs
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    for i in range(len(results["ids"][0])):  # Assuming single query
      print(f"\n🔹 Result {i + 1}")
      print(f"🆔 ID         : {results['ids'][0][i]}")
      print(f"📄 Document   : {results['documents'][0][i]}...")  # Truncate preview
      print(f"📝 Metadata   : {results['metadatas'][0][i]}")

      if "distances" in results and results["distances"][0]:
          print(f"📏 Distance   : {results['distances'][0][i]:.4f}")

      print("🔻" * 30)

    # Step 2: Extract and concatenate retrieved documents
    context_docs = results["documents"][0]
    context_text = "\n\n".join(context_docs)

    # Step 3: Build prompt
    prompt = f"""You are a helpful assistant. Use the following context to answer the question:

    Context:
    {context_text}

    Question:
    {question}

    Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate output
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=False,  # deterministic output; set to True to sample
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean answer by removing prompt prefix if needed
    # answer = answer[len(prompt):].strip()
    return answer