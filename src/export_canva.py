import csv

def compress_to_1024(values, target):
    n = len(values)
    if n <= target:
        return list(enumerate(values))   # mantiene episodios originales

    step = n // target
    compressed = []
    for i in range(0, n, step):
        compressed.append((i, values[i]))   # episodio original + valor

    return compressed[:target]               # recortar a "target"


def export_rewards_csv(mean_rewards, filename="rewards_1024.csv"):
    # Comprimir preservando episodios reales
    compressed = compress_to_1024(mean_rewards, target=100)

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(["episode", "reward"])

        for ep, val in compressed:
            writer.writerow([str(ep), float(val)])   # episodio como STRING, reward como número


def merge_reward_csvs(
    ql_file="ql_rewards_accumulated.csv",
    sarsa_file="sarsa_rewards_accumulated.csv",
    output_file="merged_rewards.csv"
):
    # Leer Q-learning
    ql_data = []
    with open(ql_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            episode = row[0]
            reward = float(row[1])
            ql_data.append((episode, reward))

    # Leer SARSA
    sarsa_data = []
    with open(sarsa_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            episode = row[0]      # lo mantenemos también como string
            reward = float(row[1])
            sarsa_data.append((episode, reward))

    # Crear CSV combinado
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(["episode", "SARSA", "Q-Learning"])

        for i in range(len(ql_data)):
            ep = ql_data[i][0]        # ya string
            ql_val = ql_data[i][1]
            sarsa_val = sarsa_data[i][1]
            writer.writerow([ep, sarsa_val, ql_val])

if __name__ == "__main__":
    merge_reward_csvs()

