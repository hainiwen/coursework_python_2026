class Fish:
    def __init__(self):
        self.members = ["Shark", "Piranha"]

    def printMembers(self):
        print("Dangerous Fish:")
        for f in self.members:
            print(f"  {f}")