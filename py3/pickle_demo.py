
def save(save_path, data):
    import pickle
    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    # import pickle; f = open("data.pickle", "wb") ; pickle.dump(data, f) ; f.close()


def load(load_path):
    import pickle
    with open(load_path, "rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data

    # import pickle ; f = open("data.pickle", "rb") ; loaded_data = pickle.load(f) ; f.close() ; print(loaded_data)



save_path = "pickle.dat"
data = ["A", "b", "C", "d"]
save(save_path, data)
loaded = load(save_path)
print(loaded)
