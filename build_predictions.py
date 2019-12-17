

with open("nlp_tarea2_preds.txt", "r") as file:
    labels = []
    for line in file:
        labels.append(line.strip())


# breakpoint()
with open("data/test_NER_esp.txt", "r") as in_file:
    with open("prediction.txt", "w") as out_file:
        labels = iter(labels)
        for line in in_file:
            if line == "\n":
                out_file.write(line)
            else:
                out_file.write(line.replace("O", next(labels)))


next(labels)
