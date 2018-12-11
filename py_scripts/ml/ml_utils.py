import dynet as dy
import sys


def create_xor_instances(num_rounds=2000):
    questions = []
    answers = []
    for round in range(num_rounds):
        for x1 in 0, 1:
            for x2 in 0, 1:
                for x3 in 0, 1:
                    answer = [0, 0, 0]
                    questions.append((x1, x2, x3))
                    answer[0] = 1
                    answers.append(tuple(answer))
    return questions, answers


def get_training_data():
    return create_xor_instances(10000)


def init_model(input_dim, hidden_num, output_dim):
    dy.renew_cg()

    model = dy.ParameterCollection()

    V = model.add_parameters((hidden_num, input_dim), init='normal')
    U = model.add_parameters((output_dim, hidden_num), init='normal')

    input_l = dy.vecInput(input_dim)

    hidden_l = dy.zeros(hidden_num)
    hidden_l = dy.logistic((V * input_l) + hidden_l)

    output_l = U * hidden_l

    return input_l, hidden_l, output_l, model


def generate_xor_training_data_to_file():
    questions, answers = create_xor_instances()
    with open('xor_training_data.txt', 'w') as f:
        for question, answer in zip(questions, answers):
            print("".join([str(q) for q in question]), "".join([str(a) for a in answer]), file=f)


def read_training_data_from_file(filename):
    with open(filename) as f:
        words = []
        word = []
        for line in f.readlines():
            if line.strip() == "":
                words.append(word)
                word = []
            else:
                word.append([[int(c) for c in elem] for elem in line.rstrip().split(' ')])
        words.append(word)

    return words


def read_from_stdin():
    words = []
    word = []
    for line in sys.stdin:
        if line.strip() == "":
            words.append(word)
            word = []
        else:
            word.append([[int(c) for c in elem] for elem in line.rstrip().split(' ')])

    return list(filter(None, words))

if __name__ == "__main__":
    generate_xor_training_data_to_file()

