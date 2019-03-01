def main():
    from skynet.mlcore.ensemble import RandomForestClassifier

    clf = RandomForestClassifier()

    print(clf)
    print(type(clf))


if __name__ == '__main__':
    main()
