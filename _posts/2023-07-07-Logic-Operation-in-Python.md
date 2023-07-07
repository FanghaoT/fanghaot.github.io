---
title: Common Mistake of Logic Operation in Python
author: fanghao
date: 2023-07-07 11:14:00 +0200
categories: [coding tips]
tags: [Python, debugging tips]
---

When it comes to logic operation in programming, I usually think about `&` or `&&`. However, it can only work as a logic operator when the data type is boolean in Python. For the other secenario, simply using `and` is the right way. If you encounter an error about datatype in a logic operation, please pay attention to this.

There is another fun fact that the `&` operator has higher precedence than the `and` operator. 