def main():
    import os
    import glob

    model_pass = '/Users/makino/PycharmProjects/SkyCC/data/ARC-common/fit_output/JMA_MSM/vis'
    backup_pass = '/Users/makino/PycharmProjects/SkyCC/data/backup'

    os.system('cp %s/*.pkl %s/' % (model_pass, backup_pass))


if __name__ == '__main__':
    main()
