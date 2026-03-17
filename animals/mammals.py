class Mammals:
    def __init__(self):
        self.members = ["Dog", "Cat", "Whale", "Dolphin"]

    def printMembers(self):
        print("Mammals:")
        for m in self.members:
            print(f"  {m}")
