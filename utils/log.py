from config import config


def log(epoch, batch_id, over, total, batch_correct_acc, total_correct_count, batch_loss, total_loss):
    print(
        "phase : {:6s}\t\tepoch : {:4.0f} \t\tbatch_id:{:3.0f}\t\ttrained : {:4.0f}/{:2.0f}\t\tbatch_loss : {:10.6f}\t\tloss:{:10.6f}\t\tbatch_correct_acc : {:1.3f}\t\tcorrect_acc : {:1.3f}".format(
            config.phase.name,
            epoch,
            batch_id,
            over,
            total,
            batch_loss,
            1.0 * total_loss / (batch_id + 1),
            batch_correct_acc ,
            1.0 * total_correct_count / over
        )
    )
