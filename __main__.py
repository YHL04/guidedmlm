

from dataset import create_pg19_data
from memory import Memory
from trainer import Trainer
from bert import BERT, Generator, Discriminator


def main():
    vocab_size = 30522
    max_len = 512
    n_layers = 4
    d_model = 512
    n_head = 8
    p = 0.1

    lr = 1e-4
    batch_size = 32

    data = create_pg19_data(max_files=1000)

    memory = Memory(
        data=data,
        dim=d_model
    )

    bert = BERT(
        vocab_size=vocab_size,
        max_len=max_len,
        n_layers=n_layers,
        d_model=d_model,
        n_head=n_head,
        p=p
    ).cuda()

    generator = Generator(
        vocab_size=vocab_size,
        max_len=max_len,
        n_layers=n_layers,
        d_model=d_model,
        n_head=n_head,
        p=p
    ).cuda()

    discriminator = Discriminator(
        vocab_size=vocab_size,
        max_len=max_len,
        n_layers=n_layers,
        d_model=d_model,
        n_head=n_head,
        p=p
    ).cuda()

    trainer = Trainer(
        bert=bert,
        generator=generator,
        discriminator=discriminator,
        memory=memory,
        lr=lr,
        batch_size=batch_size
    )

    for i in range(1000):
        loss = trainer.train_step()
        print(loss)


if __name__ == "__main__":
    main()

