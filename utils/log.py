from config import config


def log(epoch, batch_id, over, total, batch_acc, avg_acc, batch_loss, avg_loss):
    print(
        "phase : {:6s}\t\tepoch : {:4.0f} \t\tbatch_id:{:3.0f}\t\ttrained : {:4.0f}/{:2.0f}\t\tbatch_loss : {:10.6f}\t\tloss:{:10.6f}\t\tbatch_acc : {:1.3f}\t\tcorrect_acc : {:1.3f}".format(
            config.phase.name,
            epoch,
            batch_id,
            over,
            total,
            batch_loss,
            avg_loss,
            batch_acc,
            avg_acc
        )
    )
