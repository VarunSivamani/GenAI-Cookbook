def query(question: str, num_results: int = 5):
    """
    🤖 Query the ChromaDB collection for relevant chunks based on a natural language question.
    
    Args:
        question (str): 🧠 The natural language query.
        num_results (int): 🔢 Number of top results to return.
    """

    # 🛡️ Ensure collection is initialized
    if not collection:
        print("❌ [Error] No collection initialized.")
        return

    # 🔍 Display the query
    print(f'🔎 Querying collection for: "{question}"')

    try:
        # 🚀 Perform the query
        results = collection.query(
            query_texts=[question],
            n_results=num_results
        )

        # 🛡️ Handle no results
        if not results or not results.get("ids") or not results["ids"][0]:
            print("🤷‍♂️ No matching documents found.")
            return

        # 🔁 Iterate through results
        for i in range(len(results["ids"][0])):  # Assuming single query
            print(f"\n🔹 Result {i + 1}")
            print(f"🆔 ID         : {results['ids'][0][i]}")
            print(f"📄 Document   : {results['documents'][0][i][:200]}...")  # Truncate preview
            print(f"📝 Metadata   : {results['metadatas'][0][i]}")

            if "distances" in results and results["distances"][0]:
                print(f"📏 Distance   : {results['distances'][0][i]:.4f}")

            print("🔻" * 30)

    except Exception as e:
        print(f"❌ [Error] Query failed: {e}")
