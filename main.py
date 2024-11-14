from torment_nexus import TormentNexus
from pathlib import Path

try:
    data = TormentNexus.generate_data_list_from_binaries()
    train, test = TormentNexus.get_train_test_split(data)

    result_log = ""

    print("TRAIN:")
    print(train)
    result_log = result_log + train + "\n"

    print("TEST:")
    print(test)
    result_log = result_log + test + "\n"

    model = TormentNexus.get_model()
    result_log = result_log + TormentNexus.train_model(model, train, test) + "\n"
    Path("/model").mkdir(parents=True, exist_ok=True)
    TormentNexus.save_model_weights(model, "./model/gin_model_weights.pth")

    Path("/logs").mkdir(parents=True, exist_ok=True)
    with open("/logs/log.txt", "w") as text_file:
        text_file.write(result_log)
except Exception as e:
    Path("/logs").mkdir(parents=True, exist_ok=True)
    with open("/logs/error.txt", "w") as text_file:
        text_file.write(e)