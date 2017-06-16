#############################################

# 8.6 관리 속성 만들기

# 프로퍼티 정의하기

class Person:
    def __init__(self, first_name):
        self.first_name = first_name

    @property # 게터 함수 : first_name 을 프로퍼티로 만들기
    def first_name(self):
        return self._first_name

    @first_name.setter # 세터 함수 : @property 만든 이후에 추가되는 데코레이터
    def first_name(self, value):
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._first_name = value

    @first_name.deleter # 딜리ㅓ 함수 : @property 만든 이후에 추가되는 데코레이터
    def first_name(self):
        raise AttributeError("Can't delete attribute")

# 이미 존재하는 get 과 set 메소드로 프로퍼티 정의하기
class Person2:
    def __init__(self, first_name):
        self.set_first_name(first_name)

    def get_first_name(self):
        return self._first_name

    def set_first_name(self, value):
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._first_name = value

    def del_first_name(self):
        raise AttributeError("Can't delete attribute")

    name = property(get_first_name, set_first_name, del_first_name)

# 계산한 속성 정의

import math
class Circle:
    def __init__(self, radius):
        self.radius = radius

    @property
    def area(self):
        return math.pi * self.radius ** 2
    @property
    def perimeter(self):
        return 2 * math.pi * self.radius

c = Circle(4.0)
print(c.radius) # () 안 쓰고 바로 호출
print(c.perimeter)

##################################################

# 8.7 부모 클래스 메소드 호출

# 부모 클래스의 메소드 호출( super() )

class A:
    def spam(self):
        print('A.spam')

class B(A):
    def spam(self):
        print('B.spam')
        super().spam() # 부모의 spam() 호출

# __init__() 메소드 호출할 때

class A:
    def __init__(self):
        self.x = 0

class B(A):
    def __init__(self):
        super().__init__()
        self.y = 1

# 오버라이드한 코드에 사용하기

class Proxy:
    def __init__(self, obj):
        self._obj = obj

    # 내부 obj 를 위해 델리게이트 속성 찾기
    def __getattr__(self, name):
        return getattr(self._obj, name)

    # 델리게이트 속성 할당
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value) # 원본 __setattr__ 호출
        else:
            setattr(self._obj, name, value)

# 다중상속 시

class Base:
    def __init__(self):
        print('Base.__init__')

class A(Base):
    def __init__(self):
        super().__init__()
        print('A.__init__')

class B(Base):
    def __init__(self):
        super().__init__()
        print('B.__init__')

class C(A,B):
    def __init__(self):
        super().__init__()
        print('C.__init__')

###################################################

# 8.8 서브클래스에서 프로퍼티 확장

# 서브클래스에서 부모 클래스의 프로퍼티 기능 확장

class Person:
    def __init__(self, name):
        self.name = name

    @property
    def name(self):
        return self.name
    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._name = value

    @name.deleter
    def name(self):
        raise AttributeError("Can't delete attribute")

class SubPerson(Person):
    @property
    def name(self):
        print('Getting name')
        return super().name

    @name.setter
    def name(self,value):
        print('Setting name to', value)
        super(SubPerson, SubPerson).name.__set__(self,value)

    @name.deleter
    def name(self):
        print('Deleting name')
        super(SubPerson, SubPerson).name.__delete__(self)

#####################################################

# 8.9 새로운 클래스나 인스턴스 속성 만들기

class Integer:
    def __init__(self,name):
        self.name = name

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if not isinstance(value, int):
            raise TypeError('Expected an int')
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        del instance.__dict__[self.name]

class Point:
    x = Integer('x')
    y = Integer('y')
    def __init__(self, x,y):
        self.x = x
        self.y = y

# 속성 타입을 확인하는 디스크립터

class Typed :
    def __init__(self, name, expected_type):
        self.name = name
        self.expected_type = expected_type

    def __get__(self, instance, cls):
        if instance in None:
            return self
        else :
            return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError('Expected ' + str(self.expected_type))
        instance.__dict__[self.name] = value

    # 선택한 속성에 적용되는 클래스 데코레이터
    def typeassert(**kwargs):
        def decorate(cls):
            for name, expected_type in kwargs.items():
                # 클래스에 typed 디스크립터 설정
                setattr(cls, name, Typed(name, expected_type))
            return cls
        return decorate

    # 사용예
    @typeassert(name=str, shares=int, price=float)
    class Stock :
        def __init__(self, name, shares, price):
            self.name = name
            self.shares = shares
            self.price = price

#######################################################

# 8.10 게으른 계산을 하는 프로퍼티 사용

# 디스크립터 클래스

class lazyproperty:
    def __init__(self, func):
        self.func = func

    def __get_(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value

import math
class Circle :
    def __init__(self, radius):
        self.radius = radius

    @lazyproperty
    def area(self):
        print('Computing area')
        return  math.pi * self.radius ** 2

    @lazyproperty
    def perimeter(self):
        print('Computing perimeter')
        return 2 * math.pi * self.radius

# 게으르게 계산한 속성 : 성능 향상. 특정 값 사용하기 전까지 계산하지 않는 것

c = Circle(4.0)
vars(c) # 인스턴스 변수 구하기
## {'radius':4.0}

c.area # 면적 계산하고 추후 변수 확인
vars(c)

c.area # 속성에 접근해도 더 이상 프로퍼티 실행 x

del c.area # 변수 삭제하고 프로퍼티 다시 실행됨
c.area

##########################################################

# 8.11 자료 구조 초기화 단순화하기

# __init__() 함수 정의하여 단순화하기

class Structure:
    # 예상되는 필드 명시하는 클래스 변수
    _fields = []
    def __init__(self, *args):
        if len(args) != len(self._fields):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))

        # 속성 설정
        for name, value in zip(self._fields, args):
            setattr(self, name, value)

if __name__ == '__main__':
    class Stock(Structure):
        _fields = ['name', 'shares', 'price']

    class Poit(Structure):
        _fields = ['x', 'y']

    class Circle(Structure):
        _fields = ['radius']
        def area(self):
            return math.pi * self.radius * 2

class Structure:
    _fields= []
    def __init__(self, *args, **kwargs):
        if len(args) > len(self._fields):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))

        # 모든 위치 매개변수 설정
        for name, value in zip(self._fields, args):
            setattr(self, name, value)

        # 남아 있는 키워드 매개변수 설정
        for name in self._fields[len(args)]:
            setattr(self, name, kwargs.pop(name))

        # 남아 있는 기타 매개변수가 없는지 확인
        if kwargs:
            raise TypeError('Invalid argument(s): {}'.format(','.join(kwargs)))

if __name__ == '__main__':
    class Stock(Structure):
        _fields = ['name', 'shares', 'price']
    s1 = Stock('ACME', 50, 91.1)
    s2 = Stock('ACME', 50, price=91.1)
    s3 = Stock('ACME', shares=50, price=91.1)

# __init__() 메소드 일일이 작성하지 않고 쉽게 하기

class Structure:
    # 예상되는 필드를 명시하는 클래스 변수
    _fields=[]
    def __init__(self, *args):
        if len(args) != len(self._fields):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))

        # 매개변수 설정 (대안)
        self.__dict__.update(zip(self._fields, args))


