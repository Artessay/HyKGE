from multiprocessing.connection import Client

query = "胃反流和氟哌酸的关系"

client = Client(('0.0.0.0', 8207))
data_dict = {'clear': 0, 'query': query, 'function_call_type': 'KG'}
print(data_dict)
client.send(data_dict)
result = client.recv()  # Wait to receive data
client.close()
print(result)