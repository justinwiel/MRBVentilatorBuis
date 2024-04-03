import numpy as np
from PID import MovingAvgFilter

# Test case 1: Empty filter
filter1 = MovingAvgFilter(5)
assert filter1.getAvg() == 0

# Test case 2: Adding samples
filter2 = MovingAvgFilter(3)
filter2.addSample(10)
filter2.addSample(20)
filter2.addSample(30)
assert filter2.getAvg() == 20

# Test case 3: Adding more samples than sample size
filter3 = MovingAvgFilter(2)
filter3.addSample(5)
filter3.addSample(10)
filter3.addSample(15)
assert filter3.getAvg() == 12.5

# Test case 4: Adding negative samples
filter4 = MovingAvgFilter(4)
filter4.addSample(-5)
filter4.addSample(-10)
filter4.addSample(-15)
filter4.addSample(-20)
assert filter4.getAvg() == -12.5

# Test case 5: Adding floating-point samples
filter5 = MovingAvgFilter(3)
filter5.addSample(1.5)
filter5.addSample(2.5)
filter5.addSample(3.5)
assert filter5.getAvg() == 2.5

print("All test cases passed!")