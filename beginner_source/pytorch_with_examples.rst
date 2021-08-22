예시로 배우는 파이토치(PyTorch)
******************************
**Author**: `Justin Johnson <https://github.com/jcjohnson/pytorch-examples>`_

.. Note::
        이전에 제작된 파이토치(PyTorch) 튜토리얼입니다. 최신 초보자용 튜토리얼은
	`Learn the Basics <https://pytorch.org/tutorials/beginner/basics/intro.html>`_ 에서 볼 수 있습니다.



이 튜토리얼은 몇 가지 예시로 `파이토치 <https://github.com/pytorch/pytorch>`__ 의 기초가 되는 개념들을 소개합니다.  

파이토치(PyTorch)는 두 가지 핵심 내용을 제공합니다. 

- n 차원의 텐서(Tensor)는 numpy와 비슷하지만 GPU에서도 작동할 수 있습니다.
- 자동 미분(Automatic differentiation)으로 신경망을 만들고 학습합니다.

우리는 이번 예시에서 :math:`y=\sin(x)` 을 3차 다항식으로 풀어볼 것입니다. 네트워크(network)는 네 개의 매개변수(parameter)를 가질 것이고, 경사 하강법(gradient descent)으로 임의의 데이터가 만족하는 다항식을 찾기 위해 네트워크(network)의 출력(output)과 실젯값 간의 유클리디안 거리(Euclidean distance)를 최소화하여 학습됩니다.  

.. Note::
	각각의 예제들은 :ref:`이 문서의 마지막 <examples-download>` 에서 살펴볼 수 있습니다.

.. contents:: Table of Contents
	:local:

텐서(Tensors)
=======

Warm-up: numpy
--------------

파이토치(PyTorch)를 소개하기 전에 numpy로 네트워크(network)를 먼저 실행해보겠습니다. 
Numpy는 n-차원의 배열 객체를 제공하고 배열들을 다루기 위한 많은 함수들을 제공합니다. Numpy는 과학적인 컴퓨팅을 하기 위한 제네릭 프레임워크(generic framework)입니다; numpy가 계산 그래프(computation graph), 딥 러닝(deep learning), 기울기(gradients)에 대한 것을 제공하지는 않습니다. 하지만 우리는 numpy 연산을 사용한 네트워크(network)로 순전파와 역전파를 구현하여 사인(sine) 함수에 맞는 3차 다항식을 쉽게 학습할 수 있습니다. 


.. includenodoc:: /beginner/examples_tensor/polynomial_numpy.py


PyTorch: Tensors
----------------

Numpy is a great framework, but it cannot utilize GPUs to accelerate its
numerical computations. For modern deep neural networks, GPUs often
provide speedups of `50x or
greater <https://github.com/jcjohnson/cnn-benchmarks>`__, so
unfortunately numpy won't be enough for modern deep learning.

Here we introduce the most fundamental PyTorch concept: the **Tensor**.
A PyTorch Tensor is conceptually identical to a numpy array: a Tensor is
an n-dimensional array, and PyTorch provides many functions for
operating on these Tensors. Behind the scenes, Tensors can keep track of
a computational graph and gradients, but they're also useful as a
generic tool for scientific computing.

Also unlike numpy, PyTorch Tensors can utilize GPUs to accelerate
their numeric computations. To run a PyTorch Tensor on GPU, you simply
need to specify the correct device.

Here we use PyTorch Tensors to fit a third order polynomial to sine function.
Like the numpy example above we need to manually implement the forward
and backward passes through the network:

.. includenodoc:: /beginner/examples_tensor/polynomial_tensor.py


Autograd
========

PyTorch: Tensors and autograd
-------------------------------

In the above examples, we had to manually implement both the forward and
backward passes of our neural network. Manually implementing the
backward pass is not a big deal for a small two-layer network, but can
quickly get very hairy for large complex networks.

Thankfully, we can use `automatic
differentiation <https://en.wikipedia.org/wiki/Automatic_differentiation>`__
to automate the computation of backward passes in neural networks. The
**autograd** package in PyTorch provides exactly this functionality.
When using autograd, the forward pass of your network will define a
**computational graph**; nodes in the graph will be Tensors, and edges
will be functions that produce output Tensors from input Tensors.
Backpropagating through this graph then allows you to easily compute
gradients.

This sounds complicated, it's pretty simple to use in practice. Each Tensor
represents a node in a computational graph. If ``x`` is a Tensor that has
``x.requires_grad=True`` then ``x.grad`` is another Tensor holding the
gradient of ``x`` with respect to some scalar value.

Here we use PyTorch Tensors and autograd to implement our fitting sine wave
with third order polynomial example; now we no longer need to manually
implement the backward pass through the network:

.. includenodoc:: /beginner/examples_autograd/polynomial_autograd.py

PyTorch: Defining new autograd functions
----------------------------------------

Under the hood, each primitive autograd operator is really two functions
that operate on Tensors. The **forward** function computes output
Tensors from input Tensors. The **backward** function receives the
gradient of the output Tensors with respect to some scalar value, and
computes the gradient of the input Tensors with respect to that same
scalar value.

In PyTorch we can easily define our own autograd operator by defining a
subclass of ``torch.autograd.Function`` and implementing the ``forward``
and ``backward`` functions. We can then use our new autograd operator by
constructing an instance and calling it like a function, passing
Tensors containing input data.

In this example we define our model as :math:`y=a+b P_3(c+dx)` instead of
:math:`y=a+bx+cx^2+dx^3`, where :math:`P_3(x)=\frac{1}{2}\left(5x^3-3x\right)`
is the `Legendre polynomial`_ of degree three. We write our own custom autograd
function for computing forward and backward of :math:`P_3`, and use it to implement
our model:

.. _Legendre polynomial:
    https://en.wikipedia.org/wiki/Legendre_polynomials

.. includenodoc:: /beginner/examples_autograd/polynomial_custom_function.py

`nn` module
===========

PyTorch: nn
-----------

Computational graphs and autograd are a very powerful paradigm for
defining complex operators and automatically taking derivatives; however
for large neural networks raw autograd can be a bit too low-level.

When building neural networks we frequently think of arranging the
computation into **layers**, some of which have **learnable parameters**
which will be optimized during learning.

In TensorFlow, packages like
`Keras <https://github.com/fchollet/keras>`__,
`TensorFlow-Slim <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim>`__,
and `TFLearn <http://tflearn.org/>`__ provide higher-level abstractions
over raw computational graphs that are useful for building neural
networks.

In PyTorch, the ``nn`` package serves this same purpose. The ``nn``
package defines a set of **Modules**, which are roughly equivalent to
neural network layers. A Module receives input Tensors and computes
output Tensors, but may also hold internal state such as Tensors
containing learnable parameters. The ``nn`` package also defines a set
of useful loss functions that are commonly used when training neural
networks.

In this example we use the ``nn`` package to implement our polynomial model
network:

.. includenodoc:: /beginner/examples_nn/polynomial_nn.py

PyTorch: optim
--------------

Up to this point we have updated the weights of our models by manually
mutating the Tensors holding learnable parameters with ``torch.no_grad()``.
This is not a huge burden for simple optimization algorithms like stochastic
gradient descent, but in practice we often train neural networks using more
sophisticated optimizers like AdaGrad, RMSProp, Adam, etc.

The ``optim`` package in PyTorch abstracts the idea of an optimization
algorithm and provides implementations of commonly used optimization
algorithms.

In this example we will use the ``nn`` package to define our model as
before, but we will optimize the model using the RMSprop algorithm provided
by the ``optim`` package:

.. includenodoc:: /beginner/examples_nn/polynomial_optim.py

PyTorch: Custom nn Modules
--------------------------

Sometimes you will want to specify models that are more complex than a
sequence of existing Modules; for these cases you can define your own
Modules by subclassing ``nn.Module`` and defining a ``forward`` which
receives input Tensors and produces output Tensors using other
modules or other autograd operations on Tensors.

In this example we implement our third order polynomial as a custom Module
subclass:

.. includenodoc:: /beginner/examples_nn/polynomial_module.py

PyTorch: Control Flow + Weight Sharing
--------------------------------------

As an example of dynamic graphs and weight sharing, we implement a very
strange model: a third-fifth order polynomial that on each forward pass
chooses a random number between 3 and 5 and uses that many orders, reusing
the same weights multiple times to compute the fourth and fifth order.

For this model we can use normal Python flow control to implement the loop,
and we can implement weight sharing by simply reusing the same parameter multiple
times when defining the forward pass.

We can easily implement this model as a Module subclass:

.. includenodoc:: /beginner/examples_nn/dynamic_net.py


.. _examples-download:

Examples
========

You can browse the above examples here.

Tensors
-------

.. toctree::
   :maxdepth: 2
   :hidden:

   /beginner/examples_tensor/polynomial_numpy
   /beginner/examples_tensor/polynomial_tensor

.. galleryitem:: /beginner/examples_tensor/polynomial_numpy.py

.. galleryitem:: /beginner/examples_tensor/polynomial_tensor.py

.. raw:: html

    <div style='clear:both'></div>

Autograd
--------

.. toctree::
   :maxdepth: 2
   :hidden:

   /beginner/examples_autograd/polynomial_autograd
   /beginner/examples_autograd/polynomial_custom_function


.. galleryitem:: /beginner/examples_autograd/polynomial_autograd.py

.. galleryitem:: /beginner/examples_autograd/polynomial_custom_function.py

.. raw:: html

    <div style='clear:both'></div>

`nn` module
-----------

.. toctree::
   :maxdepth: 2
   :hidden:

   /beginner/examples_nn/polynomial_nn
   /beginner/examples_nn/polynomial_optim
   /beginner/examples_nn/polynomial_module
   /beginner/examples_nn/dynamic_net


.. galleryitem:: /beginner/examples_nn/polynomial_nn.py

.. galleryitem:: /beginner/examples_nn/polynomial_optim.py

.. galleryitem:: /beginner/examples_nn/polynomial_module.py

.. galleryitem:: /beginner/examples_nn/dynamic_net.py

.. raw:: html

    <div style='clear:both'></div>
