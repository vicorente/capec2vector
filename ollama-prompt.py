# Crear prompt para Ollama
                # logger.info("Preparando prompt para Ollama")
                # prompt = create_ollama_prompt(results)

                # # Realizar consulta a Ollama usando el cliente con streaming
                # response = ollama_client.chat(
                #     model="qwen2.5:72b",
                #     messages=[
                #         {
                #             "role": "system",
                #             "content": f"""
                #             Eres un experto en ciberseguridad que analiza patrones de ataque CAPEC y proporciona respuestas detalladas y útiles.
                #             Elabora las respuestas en formato markdown, con resaltado de código, listas, tablas, y aplicando saltos de linea para mejorar la legibilidad.
                #             """
                #         },
                #         {
                #             "role": "user",
                #             "content": prompt
                #         }
                #     ],
                #     stream=True,
                #     options={"temperature": 0.7}
                # )

                # # Procesar la respuesta en streaming
                # for chunk in response:
                #     if chunk and "message" in chunk and "content" in chunk["message"]:
                #         # Escapar caracteres especiales y asegurar que el JSON sea válido
                #         content = chunk["message"]["content"]
                #         try:
                #             # Intentar codificar y decodificar para asegurar que el contenido es válido
                #             content = content.encode('utf-8').decode('utf-8')
                #             data = {"chunk": content}
                #             yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                #         except Exception as e:
                #             logger.error(f"Error al procesar chunk: {str(e)}")
                #             continue