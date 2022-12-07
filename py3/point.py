from collections import namedtuple

Point = namedtuple("Point", "x y")

Point3D = namedtuple("Point3D", "x y z")

arbit = namedtuple("arbit3", "jinho ella son1 son2 daughter1 daughter2")


p1= Point(1, 3)
print(f'{p1 = }')

p2= Point3D(1, 2, 3)
print(f'{p2 = }')

p3 = arbit(1,2,3,4,5,6)
print(f'{p3 = }')
