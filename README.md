ReductionSpecializer
====================
*An example specializer built on ASPIRE's SEJITS framework*

### Objective ###

The purpose of this repository is to demonstrate the functionality of ASPIRE's SEJITS framework. This
repository contains an example specializer that performs reductions (hence the name ReductionSpecializer),
and has example code both for industry specialists who write their own specializers, as well as end-users 
who simply use the specializers. End-users write very simple Python code, gaining very efficient performance
because of our framework and the implementation of specializer-writers.

### Dependencies ###

ReductionSpecializer is based on ctree. Be sure to install ctree before attempting to run this framework 
You can find [ctree on github](http://github.com/ucb-sejits/ctree), with instructions for installation.

Additionally, ReductionSpecializer assumes that your machine has Python (any version after 2.0, but before 
3.0). This is also a prerequisite for ctree.

### Features Demonstrated ###

This example specializer demonstrates various features of ASPIRE's SEJITS framework. Below is a list of the features
that are currently included in the *master* branch of this repository.

- [X] Difference between Specializer-Writer code and End-User code
- [X] Simplicity of Python Code for end user to write
- [X] Dynamic class creation using `LazySpecializedFunction.from_function()`
- [X] Ability to have multiple `LazySpecializedFunction` subclasses use the same `ConcreteSpecializedFunction` subclass
- [X] Persistent Caching improvements (drastic speedup between the 1st run of a particular data set size and subsequent runs)
- [X] ctree syntax and usage in `LazySpecializedFunction.transform()` (in a subclass of `LazySpecializedFunction`)

### Simplicity of End-User Code ###

While the code in this repository offers a sufficient demonstration of the ease of use for the end-user, in this section
we walk a potential user through the procedure of using this specializer.

Let's say the end-user has a numpy array that we want to **sum** all the elements of. 

``` 
dataset = numpy.array([6, 2, 66, 23, 74, 23, 52, 362, 23, 723, 1042, 44, 2, 56, 27, 43])
```

Again, we want to add all those elements up. Traditionally, an end-user would have a for loop to iterate through these
elements and add them all up. However, when sizes of datasets begin to increase, this protocol becomes slow - this is
where SEJITS specializers come in. 

Now, a specializer-writer wrote a SEJITS specializer that helps facilitate an optimized reduction (we want to do a sum-reduction). So let's define the method that represents our intent.
```
def add(x, y):
    return x + y
```
Very simple; note that we're only defining a method that takes in two arguments, not the entire array. This is what is known
as a **kernel** function that we will pass into the SEJITS framework so that it knows how to operate on our `dataset`.

Now, all we have to do is make a few calls into the SEJITS specializer that has been written for us. You'll notice that in
`main.py` of this repository, `LazyRolledReduction` is one of the subclasses of `LazySpecializedFunction` (this is something
that a specializer-writer would have written). We'll use this in the following code.

```
RolledClass = LazyRolledReduction.from_function(add, "RolledClass")         # generate subclass with the add() method
sum_reducer = RolledClass()                                                 # create your reduction function, sum_reducer       
sejits_result = sum_reducer(sample_data)
```
Here, we created a reducer function (which we called `sum_reducer`) by simply dynamically creating a class called `RolledClass` that has the `add()` method that we defined earlier in it. The way we did this is the same even if we wanted to use a different
function (for example, if we had a `multiply()` function rather than an `add()` function, we could use `multiply` instead of `add` in the first line of the above code). Finally, we called the `sum_reducer` on our `sample_data` and got our output.

If we now print `sejits_results`, we get **2568**, which is the correct result. One of the best parts is that if we ever run this code again with the same data size, our caching mechanism improves performance significantly.

### Running the Sample Code ###

There are two ways of running the sample code.

The **first** is to head into terminal, change directory into where you cloned this repository and run the following command:

```
>>> python ReductionSpecializer/main.py <x> <y> <z>
```
Here, we're using `<x>`, `<y>` and `<z>` as command line arguments. Here's what they represent:

1. `<x>` is an integer that represents the GPU number. If you don't care, choose 0.
2. `<y>` is an integer that represents the work group size you want. 
3. `<z>` is an integer that represents the size of your dataset of ones.

An example of this would be:
```
>>> python ReductionSpecializer/main.py 0 32 2048
```

The **second** way to do this is to simply run the tests in the `tests.py` file. After changing into this repository's directory on your machine, simply run:
```
>>> python ReductionSpecializer/tests.py
```
