# Exercise 1: Classes and Inheritance
# =====================================
# Save this file as classroom.py and run: python classroom.py
# Or import from another script: from classroom import Student, Teacher

class Person:
    """Base class representing a person with a first and last name."""

    def __init__(self, firstname, lastname):
        self.firstname = firstname
        self.lastname = lastname

    def full_name(self):
        """Returns the full name as a combined string."""
        return f"{self.firstname} {self.lastname}"


class Student(Person):
    """A student, inheriting from Person, with a subject area."""

    def __init__(self, firstname, lastname, subject):
        super().__init__(firstname, lastname)   # call Person.__init__
        self.subject = subject

    def printNameSubject(self):
        """Prints full name and subject area."""
        print(f"{self.full_name()}, {self.subject}")


class Teacher(Person):
    """A teacher, inheriting from Person, with a course name."""

    def __init__(self, firstname, lastname, course):
        super().__init__(firstname, lastname)   # call Person.__init__
        self.course = course

    def printNameCourse(self):
        """Prints full name and course taught."""
        print(f"{self.full_name()} teaches {self.course}")


if __name__ == "__main__":
    me = Student("Benedikt", "Daurer", "physics")
    me.printNameSubject()
    # Expected output: Benedikt Daurer, physics

    print()

    t = Teacher("Ada", "Lovelace", "Python programming")
    t.printNameCourse()
    # Expected output: Ada Lovelace teaches Python programming