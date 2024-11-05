from torment_nexus import TormentNexus


data = TormentNexus.generate_data_list_from_binaries()
train, test = TormentNexus.get_train_test_split(data)


print("TRAIN:")

print(train)

print("TEST:")

print(test)