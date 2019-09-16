from time import localtime, strftime

def generate_csv_name(directory, model, epochs, seed):
    time = strftime("%Y-%m-%d %H:%M:%S", localtime())
    return (directory + time + "-" + model + "-" + str(epochs) + "-epochs-seed-" + str(seed) + ".csv")
