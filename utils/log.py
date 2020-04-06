from config import config


def log(epoch, batch_id, over, total, batch_acc, avg_acc, batch_loss, avg_loss):
    print(
        "switch : {:8s} \t\tphase : {:6s}\t\tepoch : {:4.0f} \t\tbatch_id:{:3.0f}\t\tcover : {:4.0f}/{:2.0f}\t\tbatch_loss : {:11.6f}\t\tloss:{:11.6f}\t\tbatch_acc : {:1.3f}\t\tcorrect_acc : {:1.3f}".format(
            config.weight_switch_on.name,
            config.phase.name,
            epoch + 1,
            batch_id,
            over,
            total,
            batch_loss,
            avg_loss,
            batch_acc,
            avg_acc
        )
    )
