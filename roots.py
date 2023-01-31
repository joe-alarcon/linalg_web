from fractions import Fraction

def clean_number(num, check=False):
    def check_float_return_fraction(float_num):
        assert isinstance(float_num, float)
        if round(float_num) - float_num == 0:
            return int(float_num)
        elif round(float_num*10e4)/10e4 - float_num == 0:
            return float_num
        return Fraction(float_num).limit_denominator(int(1e6))

    if isinstance(num, Root):
        rounded = round(num.get()*1e4)/1e4
        if rounded - num.get() == 0:
            return rounded
        x = clean_number(num.x, check)
        addi = clean_number(num.addi, check)
        mult = clean_number(num.multi, check)
        return type(num)(x, addi, mult)

    elif isinstance(num, float):
        if check:
            as_frac = Fraction(num).limit_denominator()
            if as_frac.denominator < 1000:
                return as_frac
            return round(num*1e4)/1e4
        else:
            return check_float_return_fraction(num)

    elif isinstance(num, Fraction):
        if num.denominator == 1:
            return num.numerator
        if check:
            return num.limit_denominator(int(1e4))
        else:
            return num.limit_denominator(int(1e8))

    elif isinstance(num, complex):
        real = clean_number(num.real, check)
        imag = clean_number(num.imag, check)
        return complex(real, imag)
    else:
        return num

def check_is_int(num):
    return round(num) - num == 0

def create_custom(power):
    def helper(x, addi=0, multi=1):
        return custom(x, power, addi, multi)
    return helper

class Root():
    def __init__(self, x, power, additive=0, multiplier=1):
        self.power = power
        self.rep = clean_number(abs(x))
        self.complex_num = isinstance(x, complex) or x < 0
        if self.complex_num:
            self.x = complex(0,clean_number(abs(x)))
            self.sq = complex(0,abs(x)**self.power)
        else:
            self.x = clean_number(x)
            self.sq = abs(x)**self.power
        self.int = check_is_int(self.sq) if not self.complex_num else False
        self.addi = clean_number(additive)
        self.multi = clean_number(multiplier)
        self.val = self.addi + self.multi*self.sq
        self.resize()

    def get(self):
        if isinstance(self.addi, Root):
            return self.multi*self.sq + self.addi.get()
        else:
            return self.val

    def resize(self):
        def add_to_dict(dict, key):
            if key in dict:
                dict[key] += 1
            else:
                dict[key] = 1

        num = self.rep
        if not check_is_int(num):
            return
        max = int(num**0.5 + 2) + 1
        dict_of_primes = {}
        for i in range(2, max):
            while num % i == 0:
                add_to_dict(dict_of_primes, i)
                num //= i
        for prime,power in dict_of_primes.items():
            reduce = prime**(power - power%(1/self.power))
            constant = prime**(power//(1/self.power))
            self.x /= reduce
            self.rep /= reduce
            self.multi *= constant
            self.sq /= constant

    def __add__(self, iother):
        other = clean_number(iother)
        if self.int:
            return self.val + other
        else:
            if isinstance(other, Root) and self.sq == other.sq:
                return Root(self.x, self.power, self.addi, self.multi + other.multi)
            return Root(self.x, self.power, self.addi + other, self.multi)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        new = self * -1
        return new.__add__(other)

    def __neg__(self):
        return -1 * self

    def __mul__(self, iother):
        if iother == 0:
            return 0
        other = clean_number(iother)
        if self.int:
            return self.get() * other
        else:
            if isinstance(other, Root) and self.power == other.power:
                a = clean_number(self.addi*other.addi)
                if self.sq == other.sq:
                    b = (self.addi*other.multi + other.addi*self.multi)*Root(other.x, self.power)
                    c = clean_number(self.multi*other.multi*self.x)
                    return a + b + c
                else:
                    b = self.addi*other.multi*Root(other.x, other.power)
                    c = other.addi*self.multi*Root(self.x, self.power)
                    d = self.multi*other.multi*Root(self.x*other.x, self.power)
                    return a + b + c + d
            elif isinstance(other, Root) and self.power != other.power:
                return self.get() * other.get()
            else:
                return Root(self.x, self.power, self.addi*other, self.multi*other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, iother):
        if iother == 0:
            raise ZeroDivisionError
        other = clean_number(iother)
        if self.int:
            return self.get() / other
        elif not isinstance(other, Root):
            new_add = clean_number(self.addi / other)
            new_mult = clean_number(self.multi / other)
            return Root(self.x, self.power, new_add, new_mult)
        elif isinstance(other, Root) and self.power == other.power and other.addi == 0:
            a = self.addi / other
            if self.sq == other.sq:
                b = clean_number(self.multi / other.multi)
                return a + b
            else:
                b = clean_number(self.multi / other.multi)*Root(clean_number(self.x / other.x), self.power)
                return a + b
        return self.get() / other.get()

    def __rtruediv__(self, other):
        if self == 0:
            raise ZeroDivisionError
        assert other is not sqrt
        if self.addi == 0:
            m = clean_number(other / (self.multi * self.x))
            return Root(self.x, self.power, multiplier=m)
        else:
            return clean_number(other / self.get())

    def __pow__(self, other):
        if self.addi == 0 and other == (1/self.power):
            if check_is_int(self.x):
                return (self.multi**other) * int(self.x)
            else:
                return (self.multi**other) * self.x
        return self.get() ** other

    def __rpow__(self, other):
        return other ** self.get()

    def __int__(self):
        return int(self.get())

    def __float__(self):
        return float(self.get())

    def __complex__(self):
        return complex(self.get())

    def __eq__(self, other):
        if isinstance(other, Root):
            return self.get() == other.get()
        else:
            return self.get() == other

    def __req__(self, other):
        return self.__eq__(other)

    def __ne__(self, other):
        if isinstance(other, Root):
            return self.get() != other.get()
        else:
            return self.get() != other

    def __rne__(self, other):
        return self.__ne__(other)

    def __lt__(self, other):
        if isinstance(other, Root):
            return self.get() < other.get()
        else:
            return self.get() < other

    def __gt__(self, other):
        if isinstance(other, Root):
            return self.get() > other.get()
        else:
            return self.get() > other

    def __le__(self, other):
        if isinstance(other, Root):
            return self.get() <= other.get()
        else:
            return self.get() <= other

    def __ge__(self, other):
        if isinstance(other, Root):
            return self.get() >= other.get()
        else:
            return self.get() >= other

    def __abs__(self):
        return self

    def __repr__(self):
        comp = "j" if self.complex_num else ""
        if self.int:
            return f"{self.val}{comp}"
        else:
            if self.multi != 1:
                sign_multi = "+" if self.multi > 0 else ""
                if self.addi == 0:
                    return f"{sign_multi}{clean_number(self.multi, True)}{comp}*({clean_number(self.rep, True)}**(1/{int(1/self.power)}))"
                else:
                    return f"{clean_number(self.addi, True)} {sign_multi}{clean_number(self.multi, True)}{comp}*({clean_number(self.rep, True)}**(1/{int(1/self.power)}))"
            else:
                if self.addi == 0:
                    return f"{comp}{clean_number(self.rep, True)}**(1/{int(1/self.power)})"
                else:
                    return f"{clean_number(self.addi, True)} + {comp}{clean_number(self.rep, True)}**(1/{int(1/self.power)})"

    def __str__(self):
        return self.__repr__()

class generic(Root):
    def __init__(self, x, power, type, addi=0, multi=1):
        super().__init__(x, power, addi, multi)
        self.type = type

    def __add__(self, other):
        result = super().__add__(other)
        return self.type(result.x, result.addi, result.multi) if isinstance(result, Root) and self.type is not Root else result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        new = self * -1
        return new.__add__(other)

    def __neg__(self):
        return -1 * self

    def __mul__(self, iother):
        result = super().__mul__(iother)
        return self.type(result.x, result.addi, result.multi) if isinstance(result, Root) and self.type is not Root else result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, iother):
        result = super().__truediv__(iother)
        return self.type(result.x, result.addi, result.multi) if isinstance(result, Root) and self.type is not Root else result

    def __rtruediv__(self, other):
        result = super().__rtruediv__(other)
        return self.type(result.x, result.addi, result.multi) if isinstance(result, Root) and self.type is not Root else result

    def __repr__(self):
        if self.type == sqrt:
            str = "sqrt"
        elif self.type == cbrt:
            str = "cbrt"
        else:
            return super(generic, self).__repr__()

        comp = "j" if self.complex_num else ""
        if self.int:
            return f"{self.val}{comp}"
        else:
            if self.multi != 1:
                sign_multi = "+" if self.multi > 0 else ""
                if self.addi == 0:
                    return f"{sign_multi}{clean_number(self.multi, True)}{comp}*{str}({clean_number(self.rep, True)})"
                else:
                    return f"{clean_number(self.addi, True)} {sign_multi}{clean_number(self.multi, True)}{comp}*{str}({clean_number(self.rep, True)})"
            else:
                if self.addi == 0:
                    return f"{comp}{str}({clean_number(self.rep, True)})"
                else:
                    return f"{clean_number(self.addi, True)} + {comp}{str}({clean_number(self.rep, True)})"


class sqrt(generic):
    def __init__(self, x, addi=0, multi=1):
        super().__init__(x, 0.5, sqrt, addi, multi)

    def conjugate(self):
        return sqrt(self.x, self.addi, -self.multi)

    def __rtruediv__(self, other):
        if not isinstance(self.addi, Root):
            conj = self.conjugate()
            denominator = self * conj
            numerator = other * conj
            return numerator / denominator
        else:
            return super().__rtruediv__(other)


class cbrt(generic):
    def __init__(self, x, addi=0, multi=1):
        super().__init__(x, 1/3, cbrt, addi, multi)


class custom(generic):
    def __init__(self, x, power, addi=0, multi=1):
        super().__init__(x, power, Root, addi, multi)
