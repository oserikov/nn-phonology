import dynet as dy


# create training instances, as before
def create_xor_instances(num_rounds=2000):
    questions = []
    answers = []
    for round in range(num_rounds):
        for x1 in 0, 1:
            for x2 in 0, 1:
                answer = 0 if x1 == x2 else 1
                questions.append((x1/2+0.5, x2/2+0.5))
                answers.append(answer)
    return questions, answers


dy.renew_cg()

INPUT_DIM = 2
HIDDEN_NUM = 2
OUTPUT_DIM = INPUT_DIM

m = dy.ParameterCollection()
V = m.add_parameters((INPUT_DIM, HIDDEN_NUM))
W = m.add_parameters((HIDDEN_NUM, HIDDEN_NUM))
U = m.add_parameters((HIDDEN_NUM, OUTPUT_DIM))
trainer = dy.SimpleSGDTrainer(m)


x = dy.vecInput(INPUT_DIM)
y = dy.scalarInput(0)


h = dy.zeros(HIDDEN_NUM)
h = dy.tanh((V * x) + (W * h))

# hidden = dy.tanh((x * V) + (hidden * W))
output = dy.softmax((U * h))
loss = dy.pick(-dy.log(output), y.npvalue())

questions, answers = create_xor_instances(10000)
seen_instances = 0
total_loss = 0
for question, answer in zip(questions, answers):
    x.set(question)
    y.set(answer)
    print(x.value())
    print(y.value())
    print(loss.scalar_value())
    loss.backward()
    trainer.update()
