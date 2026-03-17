class Birds:
    def __init__(self):
        self.members = ["Parrot", "Penguin"]

    def printMembers(self):
        print("Harmless Birds:")
        for b in self.members:
            print(f"  {b}")
