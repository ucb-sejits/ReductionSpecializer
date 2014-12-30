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

