¡Ø(
ú
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

û
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%·Ñ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.22unknown8ü´#
¬
*Adam/tic_tac_toe_neural_net/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/tic_tac_toe_neural_net/dense_8/bias/v
¥
>Adam/tic_tac_toe_neural_net/dense_8/bias/v/Read/ReadVariableOpReadVariableOp*Adam/tic_tac_toe_neural_net/dense_8/bias/v*
_output_shapes
:*
dtype0
´
,Adam/tic_tac_toe_neural_net/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*=
shared_name.,Adam/tic_tac_toe_neural_net/dense_8/kernel/v
­
@Adam/tic_tac_toe_neural_net/dense_8/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/tic_tac_toe_neural_net/dense_8/kernel/v*
_output_shapes

:(*
dtype0
¬
*Adam/tic_tac_toe_neural_net/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*;
shared_name,*Adam/tic_tac_toe_neural_net/dense_7/bias/v
¥
>Adam/tic_tac_toe_neural_net/dense_7/bias/v/Read/ReadVariableOpReadVariableOp*Adam/tic_tac_toe_neural_net/dense_7/bias/v*
_output_shapes
:(*
dtype0
´
,Adam/tic_tac_toe_neural_net/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2(*=
shared_name.,Adam/tic_tac_toe_neural_net/dense_7/kernel/v
­
@Adam/tic_tac_toe_neural_net/dense_7/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/tic_tac_toe_neural_net/dense_7/kernel/v*
_output_shapes

:2(*
dtype0
¬
*Adam/tic_tac_toe_neural_net/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*;
shared_name,*Adam/tic_tac_toe_neural_net/dense_6/bias/v
¥
>Adam/tic_tac_toe_neural_net/dense_6/bias/v/Read/ReadVariableOpReadVariableOp*Adam/tic_tac_toe_neural_net/dense_6/bias/v*
_output_shapes
:2*
dtype0
µ
,Adam/tic_tac_toe_neural_net/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*=
shared_name.,Adam/tic_tac_toe_neural_net/dense_6/kernel/v
®
@Adam/tic_tac_toe_neural_net/dense_6/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/tic_tac_toe_neural_net/dense_6/kernel/v*
_output_shapes
:	2*
dtype0
¬
*Adam/tic_tac_toe_neural_net/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*;
shared_name,*Adam/tic_tac_toe_neural_net/dense_5/bias/v
¥
>Adam/tic_tac_toe_neural_net/dense_5/bias/v/Read/ReadVariableOpReadVariableOp*Adam/tic_tac_toe_neural_net/dense_5/bias/v*
_output_shapes
:	*
dtype0
´
,Adam/tic_tac_toe_neural_net/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*=
shared_name.,Adam/tic_tac_toe_neural_net/dense_5/kernel/v
­
@Adam/tic_tac_toe_neural_net/dense_5/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/tic_tac_toe_neural_net/dense_5/kernel/v*
_output_shapes

:	*
dtype0
¬
*Adam/tic_tac_toe_neural_net/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/tic_tac_toe_neural_net/dense_4/bias/v
¥
>Adam/tic_tac_toe_neural_net/dense_4/bias/v/Read/ReadVariableOpReadVariableOp*Adam/tic_tac_toe_neural_net/dense_4/bias/v*
_output_shapes
:*
dtype0
´
,Adam/tic_tac_toe_neural_net/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*=
shared_name.,Adam/tic_tac_toe_neural_net/dense_4/kernel/v
­
@Adam/tic_tac_toe_neural_net/dense_4/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/tic_tac_toe_neural_net/dense_4/kernel/v*
_output_shapes

:*
dtype0
¬
*Adam/tic_tac_toe_neural_net/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/tic_tac_toe_neural_net/dense_3/bias/v
¥
>Adam/tic_tac_toe_neural_net/dense_3/bias/v/Read/ReadVariableOpReadVariableOp*Adam/tic_tac_toe_neural_net/dense_3/bias/v*
_output_shapes
:*
dtype0
´
,Adam/tic_tac_toe_neural_net/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*=
shared_name.,Adam/tic_tac_toe_neural_net/dense_3/kernel/v
­
@Adam/tic_tac_toe_neural_net/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/tic_tac_toe_neural_net/dense_3/kernel/v*
_output_shapes

:2*
dtype0
¬
*Adam/tic_tac_toe_neural_net/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*;
shared_name,*Adam/tic_tac_toe_neural_net/dense_2/bias/v
¥
>Adam/tic_tac_toe_neural_net/dense_2/bias/v/Read/ReadVariableOpReadVariableOp*Adam/tic_tac_toe_neural_net/dense_2/bias/v*
_output_shapes
:2*
dtype0
µ
,Adam/tic_tac_toe_neural_net/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*=
shared_name.,Adam/tic_tac_toe_neural_net/dense_2/kernel/v
®
@Adam/tic_tac_toe_neural_net/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/tic_tac_toe_neural_net/dense_2/kernel/v*
_output_shapes
:	2*
dtype0
É
8Adam/tic_tac_toe_neural_net/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/tic_tac_toe_neural_net/batch_normalization_5/beta/v
Â
LAdam/tic_tac_toe_neural_net/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp8Adam/tic_tac_toe_neural_net/batch_normalization_5/beta/v*
_output_shapes	
:*
dtype0
Ë
9Adam/tic_tac_toe_neural_net/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9Adam/tic_tac_toe_neural_net/batch_normalization_5/gamma/v
Ä
MAdam/tic_tac_toe_neural_net/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp9Adam/tic_tac_toe_neural_net/batch_normalization_5/gamma/v*
_output_shapes	
:*
dtype0
­
*Adam/tic_tac_toe_neural_net/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/tic_tac_toe_neural_net/dense_1/bias/v
¦
>Adam/tic_tac_toe_neural_net/dense_1/bias/v/Read/ReadVariableOpReadVariableOp*Adam/tic_tac_toe_neural_net/dense_1/bias/v*
_output_shapes	
:*
dtype0
¶
,Adam/tic_tac_toe_neural_net/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/tic_tac_toe_neural_net/dense_1/kernel/v
¯
@Adam/tic_tac_toe_neural_net/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/tic_tac_toe_neural_net/dense_1/kernel/v* 
_output_shapes
:
*
dtype0
É
8Adam/tic_tac_toe_neural_net/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/tic_tac_toe_neural_net/batch_normalization_4/beta/v
Â
LAdam/tic_tac_toe_neural_net/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp8Adam/tic_tac_toe_neural_net/batch_normalization_4/beta/v*
_output_shapes	
:*
dtype0
Ë
9Adam/tic_tac_toe_neural_net/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9Adam/tic_tac_toe_neural_net/batch_normalization_4/gamma/v
Ä
MAdam/tic_tac_toe_neural_net/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp9Adam/tic_tac_toe_neural_net/batch_normalization_4/gamma/v*
_output_shapes	
:*
dtype0
©
(Adam/tic_tac_toe_neural_net/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/tic_tac_toe_neural_net/dense/bias/v
¢
<Adam/tic_tac_toe_neural_net/dense/bias/v/Read/ReadVariableOpReadVariableOp(Adam/tic_tac_toe_neural_net/dense/bias/v*
_output_shapes	
:*
dtype0
²
*Adam/tic_tac_toe_neural_net/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*;
shared_name,*Adam/tic_tac_toe_neural_net/dense/kernel/v
«
>Adam/tic_tac_toe_neural_net/dense/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/tic_tac_toe_neural_net/dense/kernel/v* 
_output_shapes
:
*
dtype0
É
8Adam/tic_tac_toe_neural_net/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/tic_tac_toe_neural_net/batch_normalization_3/beta/v
Â
LAdam/tic_tac_toe_neural_net/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp8Adam/tic_tac_toe_neural_net/batch_normalization_3/beta/v*
_output_shapes	
:*
dtype0
Ë
9Adam/tic_tac_toe_neural_net/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9Adam/tic_tac_toe_neural_net/batch_normalization_3/gamma/v
Ä
MAdam/tic_tac_toe_neural_net/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp9Adam/tic_tac_toe_neural_net/batch_normalization_3/gamma/v*
_output_shapes	
:*
dtype0
¯
+Adam/tic_tac_toe_neural_net/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/tic_tac_toe_neural_net/conv2d_3/bias/v
¨
?Adam/tic_tac_toe_neural_net/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOp+Adam/tic_tac_toe_neural_net/conv2d_3/bias/v*
_output_shapes	
:*
dtype0
À
-Adam/tic_tac_toe_neural_net/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/tic_tac_toe_neural_net/conv2d_3/kernel/v
¹
AAdam/tic_tac_toe_neural_net/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOp-Adam/tic_tac_toe_neural_net/conv2d_3/kernel/v*(
_output_shapes
:*
dtype0
É
8Adam/tic_tac_toe_neural_net/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/tic_tac_toe_neural_net/batch_normalization_2/beta/v
Â
LAdam/tic_tac_toe_neural_net/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp8Adam/tic_tac_toe_neural_net/batch_normalization_2/beta/v*
_output_shapes	
:*
dtype0
Ë
9Adam/tic_tac_toe_neural_net/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9Adam/tic_tac_toe_neural_net/batch_normalization_2/gamma/v
Ä
MAdam/tic_tac_toe_neural_net/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp9Adam/tic_tac_toe_neural_net/batch_normalization_2/gamma/v*
_output_shapes	
:*
dtype0
¯
+Adam/tic_tac_toe_neural_net/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/tic_tac_toe_neural_net/conv2d_2/bias/v
¨
?Adam/tic_tac_toe_neural_net/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOp+Adam/tic_tac_toe_neural_net/conv2d_2/bias/v*
_output_shapes	
:*
dtype0
À
-Adam/tic_tac_toe_neural_net/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/tic_tac_toe_neural_net/conv2d_2/kernel/v
¹
AAdam/tic_tac_toe_neural_net/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOp-Adam/tic_tac_toe_neural_net/conv2d_2/kernel/v*(
_output_shapes
:*
dtype0
É
8Adam/tic_tac_toe_neural_net/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/tic_tac_toe_neural_net/batch_normalization_1/beta/v
Â
LAdam/tic_tac_toe_neural_net/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp8Adam/tic_tac_toe_neural_net/batch_normalization_1/beta/v*
_output_shapes	
:*
dtype0
Ë
9Adam/tic_tac_toe_neural_net/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9Adam/tic_tac_toe_neural_net/batch_normalization_1/gamma/v
Ä
MAdam/tic_tac_toe_neural_net/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp9Adam/tic_tac_toe_neural_net/batch_normalization_1/gamma/v*
_output_shapes	
:*
dtype0
¯
+Adam/tic_tac_toe_neural_net/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/tic_tac_toe_neural_net/conv2d_1/bias/v
¨
?Adam/tic_tac_toe_neural_net/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOp+Adam/tic_tac_toe_neural_net/conv2d_1/bias/v*
_output_shapes	
:*
dtype0
À
-Adam/tic_tac_toe_neural_net/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/tic_tac_toe_neural_net/conv2d_1/kernel/v
¹
AAdam/tic_tac_toe_neural_net/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOp-Adam/tic_tac_toe_neural_net/conv2d_1/kernel/v*(
_output_shapes
:*
dtype0
Å
6Adam/tic_tac_toe_neural_net/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/tic_tac_toe_neural_net/batch_normalization/beta/v
¾
JAdam/tic_tac_toe_neural_net/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOp6Adam/tic_tac_toe_neural_net/batch_normalization/beta/v*
_output_shapes	
:*
dtype0
Ç
7Adam/tic_tac_toe_neural_net/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/tic_tac_toe_neural_net/batch_normalization/gamma/v
À
KAdam/tic_tac_toe_neural_net/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp7Adam/tic_tac_toe_neural_net/batch_normalization/gamma/v*
_output_shapes	
:*
dtype0
«
)Adam/tic_tac_toe_neural_net/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/tic_tac_toe_neural_net/conv2d/bias/v
¤
=Adam/tic_tac_toe_neural_net/conv2d/bias/v/Read/ReadVariableOpReadVariableOp)Adam/tic_tac_toe_neural_net/conv2d/bias/v*
_output_shapes	
:*
dtype0
»
+Adam/tic_tac_toe_neural_net/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/tic_tac_toe_neural_net/conv2d/kernel/v
´
?Adam/tic_tac_toe_neural_net/conv2d/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/tic_tac_toe_neural_net/conv2d/kernel/v*'
_output_shapes
:*
dtype0
¬
*Adam/tic_tac_toe_neural_net/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/tic_tac_toe_neural_net/dense_8/bias/m
¥
>Adam/tic_tac_toe_neural_net/dense_8/bias/m/Read/ReadVariableOpReadVariableOp*Adam/tic_tac_toe_neural_net/dense_8/bias/m*
_output_shapes
:*
dtype0
´
,Adam/tic_tac_toe_neural_net/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*=
shared_name.,Adam/tic_tac_toe_neural_net/dense_8/kernel/m
­
@Adam/tic_tac_toe_neural_net/dense_8/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/tic_tac_toe_neural_net/dense_8/kernel/m*
_output_shapes

:(*
dtype0
¬
*Adam/tic_tac_toe_neural_net/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*;
shared_name,*Adam/tic_tac_toe_neural_net/dense_7/bias/m
¥
>Adam/tic_tac_toe_neural_net/dense_7/bias/m/Read/ReadVariableOpReadVariableOp*Adam/tic_tac_toe_neural_net/dense_7/bias/m*
_output_shapes
:(*
dtype0
´
,Adam/tic_tac_toe_neural_net/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2(*=
shared_name.,Adam/tic_tac_toe_neural_net/dense_7/kernel/m
­
@Adam/tic_tac_toe_neural_net/dense_7/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/tic_tac_toe_neural_net/dense_7/kernel/m*
_output_shapes

:2(*
dtype0
¬
*Adam/tic_tac_toe_neural_net/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*;
shared_name,*Adam/tic_tac_toe_neural_net/dense_6/bias/m
¥
>Adam/tic_tac_toe_neural_net/dense_6/bias/m/Read/ReadVariableOpReadVariableOp*Adam/tic_tac_toe_neural_net/dense_6/bias/m*
_output_shapes
:2*
dtype0
µ
,Adam/tic_tac_toe_neural_net/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*=
shared_name.,Adam/tic_tac_toe_neural_net/dense_6/kernel/m
®
@Adam/tic_tac_toe_neural_net/dense_6/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/tic_tac_toe_neural_net/dense_6/kernel/m*
_output_shapes
:	2*
dtype0
¬
*Adam/tic_tac_toe_neural_net/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*;
shared_name,*Adam/tic_tac_toe_neural_net/dense_5/bias/m
¥
>Adam/tic_tac_toe_neural_net/dense_5/bias/m/Read/ReadVariableOpReadVariableOp*Adam/tic_tac_toe_neural_net/dense_5/bias/m*
_output_shapes
:	*
dtype0
´
,Adam/tic_tac_toe_neural_net/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*=
shared_name.,Adam/tic_tac_toe_neural_net/dense_5/kernel/m
­
@Adam/tic_tac_toe_neural_net/dense_5/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/tic_tac_toe_neural_net/dense_5/kernel/m*
_output_shapes

:	*
dtype0
¬
*Adam/tic_tac_toe_neural_net/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/tic_tac_toe_neural_net/dense_4/bias/m
¥
>Adam/tic_tac_toe_neural_net/dense_4/bias/m/Read/ReadVariableOpReadVariableOp*Adam/tic_tac_toe_neural_net/dense_4/bias/m*
_output_shapes
:*
dtype0
´
,Adam/tic_tac_toe_neural_net/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*=
shared_name.,Adam/tic_tac_toe_neural_net/dense_4/kernel/m
­
@Adam/tic_tac_toe_neural_net/dense_4/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/tic_tac_toe_neural_net/dense_4/kernel/m*
_output_shapes

:*
dtype0
¬
*Adam/tic_tac_toe_neural_net/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/tic_tac_toe_neural_net/dense_3/bias/m
¥
>Adam/tic_tac_toe_neural_net/dense_3/bias/m/Read/ReadVariableOpReadVariableOp*Adam/tic_tac_toe_neural_net/dense_3/bias/m*
_output_shapes
:*
dtype0
´
,Adam/tic_tac_toe_neural_net/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*=
shared_name.,Adam/tic_tac_toe_neural_net/dense_3/kernel/m
­
@Adam/tic_tac_toe_neural_net/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/tic_tac_toe_neural_net/dense_3/kernel/m*
_output_shapes

:2*
dtype0
¬
*Adam/tic_tac_toe_neural_net/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*;
shared_name,*Adam/tic_tac_toe_neural_net/dense_2/bias/m
¥
>Adam/tic_tac_toe_neural_net/dense_2/bias/m/Read/ReadVariableOpReadVariableOp*Adam/tic_tac_toe_neural_net/dense_2/bias/m*
_output_shapes
:2*
dtype0
µ
,Adam/tic_tac_toe_neural_net/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*=
shared_name.,Adam/tic_tac_toe_neural_net/dense_2/kernel/m
®
@Adam/tic_tac_toe_neural_net/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/tic_tac_toe_neural_net/dense_2/kernel/m*
_output_shapes
:	2*
dtype0
É
8Adam/tic_tac_toe_neural_net/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/tic_tac_toe_neural_net/batch_normalization_5/beta/m
Â
LAdam/tic_tac_toe_neural_net/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp8Adam/tic_tac_toe_neural_net/batch_normalization_5/beta/m*
_output_shapes	
:*
dtype0
Ë
9Adam/tic_tac_toe_neural_net/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9Adam/tic_tac_toe_neural_net/batch_normalization_5/gamma/m
Ä
MAdam/tic_tac_toe_neural_net/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp9Adam/tic_tac_toe_neural_net/batch_normalization_5/gamma/m*
_output_shapes	
:*
dtype0
­
*Adam/tic_tac_toe_neural_net/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/tic_tac_toe_neural_net/dense_1/bias/m
¦
>Adam/tic_tac_toe_neural_net/dense_1/bias/m/Read/ReadVariableOpReadVariableOp*Adam/tic_tac_toe_neural_net/dense_1/bias/m*
_output_shapes	
:*
dtype0
¶
,Adam/tic_tac_toe_neural_net/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/tic_tac_toe_neural_net/dense_1/kernel/m
¯
@Adam/tic_tac_toe_neural_net/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/tic_tac_toe_neural_net/dense_1/kernel/m* 
_output_shapes
:
*
dtype0
É
8Adam/tic_tac_toe_neural_net/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/tic_tac_toe_neural_net/batch_normalization_4/beta/m
Â
LAdam/tic_tac_toe_neural_net/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp8Adam/tic_tac_toe_neural_net/batch_normalization_4/beta/m*
_output_shapes	
:*
dtype0
Ë
9Adam/tic_tac_toe_neural_net/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9Adam/tic_tac_toe_neural_net/batch_normalization_4/gamma/m
Ä
MAdam/tic_tac_toe_neural_net/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp9Adam/tic_tac_toe_neural_net/batch_normalization_4/gamma/m*
_output_shapes	
:*
dtype0
©
(Adam/tic_tac_toe_neural_net/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/tic_tac_toe_neural_net/dense/bias/m
¢
<Adam/tic_tac_toe_neural_net/dense/bias/m/Read/ReadVariableOpReadVariableOp(Adam/tic_tac_toe_neural_net/dense/bias/m*
_output_shapes	
:*
dtype0
²
*Adam/tic_tac_toe_neural_net/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*;
shared_name,*Adam/tic_tac_toe_neural_net/dense/kernel/m
«
>Adam/tic_tac_toe_neural_net/dense/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/tic_tac_toe_neural_net/dense/kernel/m* 
_output_shapes
:
*
dtype0
É
8Adam/tic_tac_toe_neural_net/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/tic_tac_toe_neural_net/batch_normalization_3/beta/m
Â
LAdam/tic_tac_toe_neural_net/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp8Adam/tic_tac_toe_neural_net/batch_normalization_3/beta/m*
_output_shapes	
:*
dtype0
Ë
9Adam/tic_tac_toe_neural_net/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9Adam/tic_tac_toe_neural_net/batch_normalization_3/gamma/m
Ä
MAdam/tic_tac_toe_neural_net/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp9Adam/tic_tac_toe_neural_net/batch_normalization_3/gamma/m*
_output_shapes	
:*
dtype0
¯
+Adam/tic_tac_toe_neural_net/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/tic_tac_toe_neural_net/conv2d_3/bias/m
¨
?Adam/tic_tac_toe_neural_net/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOp+Adam/tic_tac_toe_neural_net/conv2d_3/bias/m*
_output_shapes	
:*
dtype0
À
-Adam/tic_tac_toe_neural_net/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/tic_tac_toe_neural_net/conv2d_3/kernel/m
¹
AAdam/tic_tac_toe_neural_net/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOp-Adam/tic_tac_toe_neural_net/conv2d_3/kernel/m*(
_output_shapes
:*
dtype0
É
8Adam/tic_tac_toe_neural_net/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/tic_tac_toe_neural_net/batch_normalization_2/beta/m
Â
LAdam/tic_tac_toe_neural_net/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp8Adam/tic_tac_toe_neural_net/batch_normalization_2/beta/m*
_output_shapes	
:*
dtype0
Ë
9Adam/tic_tac_toe_neural_net/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9Adam/tic_tac_toe_neural_net/batch_normalization_2/gamma/m
Ä
MAdam/tic_tac_toe_neural_net/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp9Adam/tic_tac_toe_neural_net/batch_normalization_2/gamma/m*
_output_shapes	
:*
dtype0
¯
+Adam/tic_tac_toe_neural_net/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/tic_tac_toe_neural_net/conv2d_2/bias/m
¨
?Adam/tic_tac_toe_neural_net/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOp+Adam/tic_tac_toe_neural_net/conv2d_2/bias/m*
_output_shapes	
:*
dtype0
À
-Adam/tic_tac_toe_neural_net/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/tic_tac_toe_neural_net/conv2d_2/kernel/m
¹
AAdam/tic_tac_toe_neural_net/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOp-Adam/tic_tac_toe_neural_net/conv2d_2/kernel/m*(
_output_shapes
:*
dtype0
É
8Adam/tic_tac_toe_neural_net/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/tic_tac_toe_neural_net/batch_normalization_1/beta/m
Â
LAdam/tic_tac_toe_neural_net/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp8Adam/tic_tac_toe_neural_net/batch_normalization_1/beta/m*
_output_shapes	
:*
dtype0
Ë
9Adam/tic_tac_toe_neural_net/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9Adam/tic_tac_toe_neural_net/batch_normalization_1/gamma/m
Ä
MAdam/tic_tac_toe_neural_net/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp9Adam/tic_tac_toe_neural_net/batch_normalization_1/gamma/m*
_output_shapes	
:*
dtype0
¯
+Adam/tic_tac_toe_neural_net/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/tic_tac_toe_neural_net/conv2d_1/bias/m
¨
?Adam/tic_tac_toe_neural_net/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOp+Adam/tic_tac_toe_neural_net/conv2d_1/bias/m*
_output_shapes	
:*
dtype0
À
-Adam/tic_tac_toe_neural_net/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/tic_tac_toe_neural_net/conv2d_1/kernel/m
¹
AAdam/tic_tac_toe_neural_net/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOp-Adam/tic_tac_toe_neural_net/conv2d_1/kernel/m*(
_output_shapes
:*
dtype0
Å
6Adam/tic_tac_toe_neural_net/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/tic_tac_toe_neural_net/batch_normalization/beta/m
¾
JAdam/tic_tac_toe_neural_net/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOp6Adam/tic_tac_toe_neural_net/batch_normalization/beta/m*
_output_shapes	
:*
dtype0
Ç
7Adam/tic_tac_toe_neural_net/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/tic_tac_toe_neural_net/batch_normalization/gamma/m
À
KAdam/tic_tac_toe_neural_net/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp7Adam/tic_tac_toe_neural_net/batch_normalization/gamma/m*
_output_shapes	
:*
dtype0
«
)Adam/tic_tac_toe_neural_net/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/tic_tac_toe_neural_net/conv2d/bias/m
¤
=Adam/tic_tac_toe_neural_net/conv2d/bias/m/Read/ReadVariableOpReadVariableOp)Adam/tic_tac_toe_neural_net/conv2d/bias/m*
_output_shapes	
:*
dtype0
»
+Adam/tic_tac_toe_neural_net/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/tic_tac_toe_neural_net/conv2d/kernel/m
´
?Adam/tic_tac_toe_neural_net/conv2d/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/tic_tac_toe_neural_net/conv2d/kernel/m*'
_output_shapes
:*
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0

#tic_tac_toe_neural_net/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#tic_tac_toe_neural_net/dense_8/bias

7tic_tac_toe_neural_net/dense_8/bias/Read/ReadVariableOpReadVariableOp#tic_tac_toe_neural_net/dense_8/bias*
_output_shapes
:*
dtype0
¦
%tic_tac_toe_neural_net/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*6
shared_name'%tic_tac_toe_neural_net/dense_8/kernel

9tic_tac_toe_neural_net/dense_8/kernel/Read/ReadVariableOpReadVariableOp%tic_tac_toe_neural_net/dense_8/kernel*
_output_shapes

:(*
dtype0

#tic_tac_toe_neural_net/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*4
shared_name%#tic_tac_toe_neural_net/dense_7/bias

7tic_tac_toe_neural_net/dense_7/bias/Read/ReadVariableOpReadVariableOp#tic_tac_toe_neural_net/dense_7/bias*
_output_shapes
:(*
dtype0
¦
%tic_tac_toe_neural_net/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2(*6
shared_name'%tic_tac_toe_neural_net/dense_7/kernel

9tic_tac_toe_neural_net/dense_7/kernel/Read/ReadVariableOpReadVariableOp%tic_tac_toe_neural_net/dense_7/kernel*
_output_shapes

:2(*
dtype0

#tic_tac_toe_neural_net/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#tic_tac_toe_neural_net/dense_6/bias

7tic_tac_toe_neural_net/dense_6/bias/Read/ReadVariableOpReadVariableOp#tic_tac_toe_neural_net/dense_6/bias*
_output_shapes
:2*
dtype0
§
%tic_tac_toe_neural_net/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*6
shared_name'%tic_tac_toe_neural_net/dense_6/kernel
 
9tic_tac_toe_neural_net/dense_6/kernel/Read/ReadVariableOpReadVariableOp%tic_tac_toe_neural_net/dense_6/kernel*
_output_shapes
:	2*
dtype0

#tic_tac_toe_neural_net/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*4
shared_name%#tic_tac_toe_neural_net/dense_5/bias

7tic_tac_toe_neural_net/dense_5/bias/Read/ReadVariableOpReadVariableOp#tic_tac_toe_neural_net/dense_5/bias*
_output_shapes
:	*
dtype0
¦
%tic_tac_toe_neural_net/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*6
shared_name'%tic_tac_toe_neural_net/dense_5/kernel

9tic_tac_toe_neural_net/dense_5/kernel/Read/ReadVariableOpReadVariableOp%tic_tac_toe_neural_net/dense_5/kernel*
_output_shapes

:	*
dtype0

#tic_tac_toe_neural_net/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#tic_tac_toe_neural_net/dense_4/bias

7tic_tac_toe_neural_net/dense_4/bias/Read/ReadVariableOpReadVariableOp#tic_tac_toe_neural_net/dense_4/bias*
_output_shapes
:*
dtype0
¦
%tic_tac_toe_neural_net/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%tic_tac_toe_neural_net/dense_4/kernel

9tic_tac_toe_neural_net/dense_4/kernel/Read/ReadVariableOpReadVariableOp%tic_tac_toe_neural_net/dense_4/kernel*
_output_shapes

:*
dtype0

#tic_tac_toe_neural_net/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#tic_tac_toe_neural_net/dense_3/bias

7tic_tac_toe_neural_net/dense_3/bias/Read/ReadVariableOpReadVariableOp#tic_tac_toe_neural_net/dense_3/bias*
_output_shapes
:*
dtype0
¦
%tic_tac_toe_neural_net/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*6
shared_name'%tic_tac_toe_neural_net/dense_3/kernel

9tic_tac_toe_neural_net/dense_3/kernel/Read/ReadVariableOpReadVariableOp%tic_tac_toe_neural_net/dense_3/kernel*
_output_shapes

:2*
dtype0

#tic_tac_toe_neural_net/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*4
shared_name%#tic_tac_toe_neural_net/dense_2/bias

7tic_tac_toe_neural_net/dense_2/bias/Read/ReadVariableOpReadVariableOp#tic_tac_toe_neural_net/dense_2/bias*
_output_shapes
:2*
dtype0
§
%tic_tac_toe_neural_net/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*6
shared_name'%tic_tac_toe_neural_net/dense_2/kernel
 
9tic_tac_toe_neural_net/dense_2/kernel/Read/ReadVariableOpReadVariableOp%tic_tac_toe_neural_net/dense_2/kernel*
_output_shapes
:	2*
dtype0
Ñ
<tic_tac_toe_neural_net/batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><tic_tac_toe_neural_net/batch_normalization_5/moving_variance
Ê
Ptic_tac_toe_neural_net/batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp<tic_tac_toe_neural_net/batch_normalization_5/moving_variance*
_output_shapes	
:*
dtype0
É
8tic_tac_toe_neural_net/batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8tic_tac_toe_neural_net/batch_normalization_5/moving_mean
Â
Ltic_tac_toe_neural_net/batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp8tic_tac_toe_neural_net/batch_normalization_5/moving_mean*
_output_shapes	
:*
dtype0
»
1tic_tac_toe_neural_net/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31tic_tac_toe_neural_net/batch_normalization_5/beta
´
Etic_tac_toe_neural_net/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp1tic_tac_toe_neural_net/batch_normalization_5/beta*
_output_shapes	
:*
dtype0
½
2tic_tac_toe_neural_net/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42tic_tac_toe_neural_net/batch_normalization_5/gamma
¶
Ftic_tac_toe_neural_net/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp2tic_tac_toe_neural_net/batch_normalization_5/gamma*
_output_shapes	
:*
dtype0

#tic_tac_toe_neural_net/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#tic_tac_toe_neural_net/dense_1/bias

7tic_tac_toe_neural_net/dense_1/bias/Read/ReadVariableOpReadVariableOp#tic_tac_toe_neural_net/dense_1/bias*
_output_shapes	
:*
dtype0
¨
%tic_tac_toe_neural_net/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%tic_tac_toe_neural_net/dense_1/kernel
¡
9tic_tac_toe_neural_net/dense_1/kernel/Read/ReadVariableOpReadVariableOp%tic_tac_toe_neural_net/dense_1/kernel* 
_output_shapes
:
*
dtype0
Ñ
<tic_tac_toe_neural_net/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><tic_tac_toe_neural_net/batch_normalization_4/moving_variance
Ê
Ptic_tac_toe_neural_net/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp<tic_tac_toe_neural_net/batch_normalization_4/moving_variance*
_output_shapes	
:*
dtype0
É
8tic_tac_toe_neural_net/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8tic_tac_toe_neural_net/batch_normalization_4/moving_mean
Â
Ltic_tac_toe_neural_net/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp8tic_tac_toe_neural_net/batch_normalization_4/moving_mean*
_output_shapes	
:*
dtype0
»
1tic_tac_toe_neural_net/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31tic_tac_toe_neural_net/batch_normalization_4/beta
´
Etic_tac_toe_neural_net/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp1tic_tac_toe_neural_net/batch_normalization_4/beta*
_output_shapes	
:*
dtype0
½
2tic_tac_toe_neural_net/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42tic_tac_toe_neural_net/batch_normalization_4/gamma
¶
Ftic_tac_toe_neural_net/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp2tic_tac_toe_neural_net/batch_normalization_4/gamma*
_output_shapes	
:*
dtype0

!tic_tac_toe_neural_net/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!tic_tac_toe_neural_net/dense/bias

5tic_tac_toe_neural_net/dense/bias/Read/ReadVariableOpReadVariableOp!tic_tac_toe_neural_net/dense/bias*
_output_shapes	
:*
dtype0
¤
#tic_tac_toe_neural_net/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#tic_tac_toe_neural_net/dense/kernel

7tic_tac_toe_neural_net/dense/kernel/Read/ReadVariableOpReadVariableOp#tic_tac_toe_neural_net/dense/kernel* 
_output_shapes
:
*
dtype0
Ñ
<tic_tac_toe_neural_net/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><tic_tac_toe_neural_net/batch_normalization_3/moving_variance
Ê
Ptic_tac_toe_neural_net/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp<tic_tac_toe_neural_net/batch_normalization_3/moving_variance*
_output_shapes	
:*
dtype0
É
8tic_tac_toe_neural_net/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8tic_tac_toe_neural_net/batch_normalization_3/moving_mean
Â
Ltic_tac_toe_neural_net/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp8tic_tac_toe_neural_net/batch_normalization_3/moving_mean*
_output_shapes	
:*
dtype0
»
1tic_tac_toe_neural_net/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31tic_tac_toe_neural_net/batch_normalization_3/beta
´
Etic_tac_toe_neural_net/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp1tic_tac_toe_neural_net/batch_normalization_3/beta*
_output_shapes	
:*
dtype0
½
2tic_tac_toe_neural_net/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42tic_tac_toe_neural_net/batch_normalization_3/gamma
¶
Ftic_tac_toe_neural_net/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp2tic_tac_toe_neural_net/batch_normalization_3/gamma*
_output_shapes	
:*
dtype0
¡
$tic_tac_toe_neural_net/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$tic_tac_toe_neural_net/conv2d_3/bias

8tic_tac_toe_neural_net/conv2d_3/bias/Read/ReadVariableOpReadVariableOp$tic_tac_toe_neural_net/conv2d_3/bias*
_output_shapes	
:*
dtype0
²
&tic_tac_toe_neural_net/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&tic_tac_toe_neural_net/conv2d_3/kernel
«
:tic_tac_toe_neural_net/conv2d_3/kernel/Read/ReadVariableOpReadVariableOp&tic_tac_toe_neural_net/conv2d_3/kernel*(
_output_shapes
:*
dtype0
Ñ
<tic_tac_toe_neural_net/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><tic_tac_toe_neural_net/batch_normalization_2/moving_variance
Ê
Ptic_tac_toe_neural_net/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp<tic_tac_toe_neural_net/batch_normalization_2/moving_variance*
_output_shapes	
:*
dtype0
É
8tic_tac_toe_neural_net/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8tic_tac_toe_neural_net/batch_normalization_2/moving_mean
Â
Ltic_tac_toe_neural_net/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp8tic_tac_toe_neural_net/batch_normalization_2/moving_mean*
_output_shapes	
:*
dtype0
»
1tic_tac_toe_neural_net/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31tic_tac_toe_neural_net/batch_normalization_2/beta
´
Etic_tac_toe_neural_net/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp1tic_tac_toe_neural_net/batch_normalization_2/beta*
_output_shapes	
:*
dtype0
½
2tic_tac_toe_neural_net/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42tic_tac_toe_neural_net/batch_normalization_2/gamma
¶
Ftic_tac_toe_neural_net/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp2tic_tac_toe_neural_net/batch_normalization_2/gamma*
_output_shapes	
:*
dtype0
¡
$tic_tac_toe_neural_net/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$tic_tac_toe_neural_net/conv2d_2/bias

8tic_tac_toe_neural_net/conv2d_2/bias/Read/ReadVariableOpReadVariableOp$tic_tac_toe_neural_net/conv2d_2/bias*
_output_shapes	
:*
dtype0
²
&tic_tac_toe_neural_net/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&tic_tac_toe_neural_net/conv2d_2/kernel
«
:tic_tac_toe_neural_net/conv2d_2/kernel/Read/ReadVariableOpReadVariableOp&tic_tac_toe_neural_net/conv2d_2/kernel*(
_output_shapes
:*
dtype0
Ñ
<tic_tac_toe_neural_net/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><tic_tac_toe_neural_net/batch_normalization_1/moving_variance
Ê
Ptic_tac_toe_neural_net/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp<tic_tac_toe_neural_net/batch_normalization_1/moving_variance*
_output_shapes	
:*
dtype0
É
8tic_tac_toe_neural_net/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8tic_tac_toe_neural_net/batch_normalization_1/moving_mean
Â
Ltic_tac_toe_neural_net/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp8tic_tac_toe_neural_net/batch_normalization_1/moving_mean*
_output_shapes	
:*
dtype0
»
1tic_tac_toe_neural_net/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31tic_tac_toe_neural_net/batch_normalization_1/beta
´
Etic_tac_toe_neural_net/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp1tic_tac_toe_neural_net/batch_normalization_1/beta*
_output_shapes	
:*
dtype0
½
2tic_tac_toe_neural_net/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42tic_tac_toe_neural_net/batch_normalization_1/gamma
¶
Ftic_tac_toe_neural_net/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp2tic_tac_toe_neural_net/batch_normalization_1/gamma*
_output_shapes	
:*
dtype0
¡
$tic_tac_toe_neural_net/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$tic_tac_toe_neural_net/conv2d_1/bias

8tic_tac_toe_neural_net/conv2d_1/bias/Read/ReadVariableOpReadVariableOp$tic_tac_toe_neural_net/conv2d_1/bias*
_output_shapes	
:*
dtype0
²
&tic_tac_toe_neural_net/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&tic_tac_toe_neural_net/conv2d_1/kernel
«
:tic_tac_toe_neural_net/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp&tic_tac_toe_neural_net/conv2d_1/kernel*(
_output_shapes
:*
dtype0
Í
:tic_tac_toe_neural_net/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:tic_tac_toe_neural_net/batch_normalization/moving_variance
Æ
Ntic_tac_toe_neural_net/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp:tic_tac_toe_neural_net/batch_normalization/moving_variance*
_output_shapes	
:*
dtype0
Å
6tic_tac_toe_neural_net/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86tic_tac_toe_neural_net/batch_normalization/moving_mean
¾
Jtic_tac_toe_neural_net/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp6tic_tac_toe_neural_net/batch_normalization/moving_mean*
_output_shapes	
:*
dtype0
·
/tic_tac_toe_neural_net/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/tic_tac_toe_neural_net/batch_normalization/beta
°
Ctic_tac_toe_neural_net/batch_normalization/beta/Read/ReadVariableOpReadVariableOp/tic_tac_toe_neural_net/batch_normalization/beta*
_output_shapes	
:*
dtype0
¹
0tic_tac_toe_neural_net/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20tic_tac_toe_neural_net/batch_normalization/gamma
²
Dtic_tac_toe_neural_net/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp0tic_tac_toe_neural_net/batch_normalization/gamma*
_output_shapes	
:*
dtype0

"tic_tac_toe_neural_net/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"tic_tac_toe_neural_net/conv2d/bias

6tic_tac_toe_neural_net/conv2d/bias/Read/ReadVariableOpReadVariableOp"tic_tac_toe_neural_net/conv2d/bias*
_output_shapes	
:*
dtype0
­
$tic_tac_toe_neural_net/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$tic_tac_toe_neural_net/conv2d/kernel
¦
8tic_tac_toe_neural_net/conv2d/kernel/Read/ReadVariableOpReadVariableOp$tic_tac_toe_neural_net/conv2d/kernel*'
_output_shapes
:*
dtype0

NoOpNoOp
î
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¨
valueB B

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
L1_conv
	L1_conv_batch_norm

L2_conv
L2_conv_batch_norm
L3_conv
L3_conv_batch_norm
L4_conv
L4_conv_batch_norm

L1_flatten
L1_dense
L1_dense_batch_norm
L2_dense
L2_dense_batch_norm
dropout
L1_policy_dense
L2_policy_dense
L3_policy_dense

policy_out
L1_value
L2_value
	value_out
	optimizer
policy_loss_metric
value_loss_metric
 loss_metric
!
signatures*
º
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519
620
721
822
923
:24
;25
<26
=27
>28
?29
@30
A31
B32
C33
D34
E35
F36
G37
H38
I39
J40
K41
L42
M43
N44
O45
P46
Q47
R48
S49
T50
U51
V52
W53
X54
Y55*
ª
"0
#1
$2
%3
(4
)5
*6
+7
.8
/9
010
111
412
513
614
715
:16
;17
<18
=19
@20
A21
B22
C23
F24
G25
H26
I27
J28
K29
L30
M31
N32
O33
P34
Q35
R36
S37*
* 
°
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
_trace_0
`trace_1
atrace_2
btrace_3* 
6
ctrace_0
dtrace_1
etrace_2
ftrace_3* 
* 
È
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses

"kernel
#bias
 m_jit_compiled_convolution_op*
Õ
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses
taxis
	$gamma
%beta
&moving_mean
'moving_variance*
È
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses

(kernel
)bias
 {_jit_compiled_convolution_op*
Ø
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis
	*gamma
+beta
,moving_mean
-moving_variance*
Ï
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

.kernel
/bias
!_jit_compiled_convolution_op*
Ü
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis
	0gamma
1beta
2moving_mean
3moving_variance*
Ï
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

4kernel
5bias
!_jit_compiled_convolution_op*
Ü
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis
	6gamma
7beta
8moving_mean
9moving_variance*

	variables
 trainable_variables
¡regularization_losses
¢	keras_api
£__call__
+¤&call_and_return_all_conditional_losses* 
¬
¥	variables
¦trainable_variables
§regularization_losses
¨	keras_api
©__call__
+ª&call_and_return_all_conditional_losses

:kernel
;bias*
Ü
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
¯__call__
+°&call_and_return_all_conditional_losses
	±axis
	<gamma
=beta
>moving_mean
?moving_variance*
¬
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
¶__call__
+·&call_and_return_all_conditional_losses

@kernel
Abias*
Ü
¸	variables
¹trainable_variables
ºregularization_losses
»	keras_api
¼__call__
+½&call_and_return_all_conditional_losses
	¾axis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance*
¬
¿	variables
Àtrainable_variables
Áregularization_losses
Â	keras_api
Ã__call__
+Ä&call_and_return_all_conditional_losses
Å_random_generator* 
¬
Æ	variables
Çtrainable_variables
Èregularization_losses
É	keras_api
Ê__call__
+Ë&call_and_return_all_conditional_losses

Fkernel
Gbias*
¬
Ì	variables
Ítrainable_variables
Îregularization_losses
Ï	keras_api
Ð__call__
+Ñ&call_and_return_all_conditional_losses

Hkernel
Ibias*
¬
Ò	variables
Ótrainable_variables
Ôregularization_losses
Õ	keras_api
Ö__call__
+×&call_and_return_all_conditional_losses

Jkernel
Kbias*
¬
Ø	variables
Ùtrainable_variables
Úregularization_losses
Û	keras_api
Ü__call__
+Ý&call_and_return_all_conditional_losses

Lkernel
Mbias*
¬
Þ	variables
ßtrainable_variables
àregularization_losses
á	keras_api
â__call__
+ã&call_and_return_all_conditional_losses

Nkernel
Obias*
¬
ä	variables
åtrainable_variables
æregularization_losses
ç	keras_api
è__call__
+é&call_and_return_all_conditional_losses

Pkernel
Qbias*
¬
ê	variables
ëtrainable_variables
ìregularization_losses
í	keras_api
î__call__
+ï&call_and_return_all_conditional_losses

Rkernel
Sbias*
­
	ðiter
ñbeta_1
òbeta_2

ódecay"m°#m±$m²%m³(m´)mµ*m¶+m·.m¸/m¹0mº1m»4m¼5m½6m¾7m¿:mÀ;mÁ<mÂ=mÃ@mÄAmÅBmÆCmÇFmÈGmÉHmÊImËJmÌKmÍLmÎMmÏNmÐOmÑPmÒQmÓRmÔSmÕ"vÖ#v×$vØ%vÙ(vÚ)vÛ*vÜ+vÝ.vÞ/vß0và1vá4vâ5vã6vä7vå:væ;vç<vè=vé@vêAvëBvìCvíFvîGvïHvðIvñJvòKvóLvôMvõNvöOv÷PvøQvùRvúSvû*
:
ô	variables
õ	keras_api
	Ttotal
	Ucount*
:
ö	variables
÷	keras_api
	Vtotal
	Wcount*
:
ø	variables
ù	keras_api
	Xtotal
	Ycount*

úserving_default* 
d^
VARIABLE_VALUE$tic_tac_toe_neural_net/conv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"tic_tac_toe_neural_net/conv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE0tic_tac_toe_neural_net/batch_normalization/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/tic_tac_toe_neural_net/batch_normalization/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE6tic_tac_toe_neural_net/batch_normalization/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE:tic_tac_toe_neural_net/batch_normalization/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&tic_tac_toe_neural_net/conv2d_1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$tic_tac_toe_neural_net/conv2d_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE2tic_tac_toe_neural_net/batch_normalization_1/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1tic_tac_toe_neural_net/batch_normalization_1/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8tic_tac_toe_neural_net/batch_normalization_1/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<tic_tac_toe_neural_net/batch_normalization_1/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&tic_tac_toe_neural_net/conv2d_2/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$tic_tac_toe_neural_net/conv2d_2/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2tic_tac_toe_neural_net/batch_normalization_2/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1tic_tac_toe_neural_net/batch_normalization_2/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8tic_tac_toe_neural_net/batch_normalization_2/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<tic_tac_toe_neural_net/batch_normalization_2/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&tic_tac_toe_neural_net/conv2d_3/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$tic_tac_toe_neural_net/conv2d_3/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2tic_tac_toe_neural_net/batch_normalization_3/gamma'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1tic_tac_toe_neural_net/batch_normalization_3/beta'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8tic_tac_toe_neural_net/batch_normalization_3/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<tic_tac_toe_neural_net/batch_normalization_3/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#tic_tac_toe_neural_net/dense/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!tic_tac_toe_neural_net/dense/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2tic_tac_toe_neural_net/batch_normalization_4/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1tic_tac_toe_neural_net/batch_normalization_4/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8tic_tac_toe_neural_net/batch_normalization_4/moving_mean'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<tic_tac_toe_neural_net/batch_normalization_4/moving_variance'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%tic_tac_toe_neural_net/dense_1/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#tic_tac_toe_neural_net/dense_1/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2tic_tac_toe_neural_net/batch_normalization_5/gamma'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1tic_tac_toe_neural_net/batch_normalization_5/beta'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8tic_tac_toe_neural_net/batch_normalization_5/moving_mean'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<tic_tac_toe_neural_net/batch_normalization_5/moving_variance'variables/35/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%tic_tac_toe_neural_net/dense_2/kernel'variables/36/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#tic_tac_toe_neural_net/dense_2/bias'variables/37/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%tic_tac_toe_neural_net/dense_3/kernel'variables/38/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#tic_tac_toe_neural_net/dense_3/bias'variables/39/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%tic_tac_toe_neural_net/dense_4/kernel'variables/40/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#tic_tac_toe_neural_net/dense_4/bias'variables/41/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%tic_tac_toe_neural_net/dense_5/kernel'variables/42/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#tic_tac_toe_neural_net/dense_5/bias'variables/43/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%tic_tac_toe_neural_net/dense_6/kernel'variables/44/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#tic_tac_toe_neural_net/dense_6/bias'variables/45/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%tic_tac_toe_neural_net/dense_7/kernel'variables/46/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#tic_tac_toe_neural_net/dense_7/bias'variables/47/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%tic_tac_toe_neural_net/dense_8/kernel'variables/48/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#tic_tac_toe_neural_net/dense_8/bias'variables/49/.ATTRIBUTES/VARIABLE_VALUE*
HB
VARIABLE_VALUEtotal_2'variables/50/.ATTRIBUTES/VARIABLE_VALUE*
HB
VARIABLE_VALUEcount_2'variables/51/.ATTRIBUTES/VARIABLE_VALUE*
HB
VARIABLE_VALUEtotal_1'variables/52/.ATTRIBUTES/VARIABLE_VALUE*
HB
VARIABLE_VALUEcount_1'variables/53/.ATTRIBUTES/VARIABLE_VALUE*
F@
VARIABLE_VALUEtotal'variables/54/.ATTRIBUTES/VARIABLE_VALUE*
F@
VARIABLE_VALUEcount'variables/55/.ATTRIBUTES/VARIABLE_VALUE*

&0
'1
,2
-3
24
35
86
97
>8
?9
D10
E11
T12
U13
V14
W15
X16
Y17*
¢
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20*

0
1
 2*
* 
A
total_policy_loss
total_value_loss
 
total_loss*
* 
* 
* 
* 
* 
* 
* 
* 

"0
#1*

"0
#1*
* 

ûnon_trainable_variables
ülayers
ýmetrics
 þlayer_regularization_losses
ÿlayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
 
$0
%1
&2
'3*

$0
%1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 

(0
)1*

(0
)1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
 
*0
+1
,2
-3*

*0
+1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 

.0
/1*

.0
/1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

 trace_0* 

¡trace_0* 
* 
 
00
11
22
33*

00
11*
* 

¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

§trace_0
¨trace_1* 

©trace_0
ªtrace_1* 
* 

40
51*

40
51*
* 

«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
¯layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

°trace_0* 

±trace_0* 
* 
 
60
71
82
93*

60
71*
* 

²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

·trace_0
¸trace_1* 

¹trace_0
ºtrace_1* 
* 
* 
* 
* 

»non_trainable_variables
¼layers
½metrics
 ¾layer_regularization_losses
¿layer_metrics
	variables
 trainable_variables
¡regularization_losses
£__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses* 

Àtrace_0* 

Átrace_0* 

:0
;1*

:0
;1*
* 

Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
¥	variables
¦trainable_variables
§regularization_losses
©__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses*

Çtrace_0* 

Ètrace_0* 
 
<0
=1
>2
?3*

<0
=1*
* 

Énon_trainable_variables
Êlayers
Ëmetrics
 Ìlayer_regularization_losses
Ílayer_metrics
«	variables
¬trainable_variables
­regularization_losses
¯__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses*

Îtrace_0
Ïtrace_1* 

Ðtrace_0
Ñtrace_1* 
* 

@0
A1*

@0
A1*
* 

Ònon_trainable_variables
Ólayers
Ômetrics
 Õlayer_regularization_losses
Ölayer_metrics
²	variables
³trainable_variables
´regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses*

×trace_0* 

Øtrace_0* 
 
B0
C1
D2
E3*

B0
C1*
* 

Ùnon_trainable_variables
Úlayers
Ûmetrics
 Ülayer_regularization_losses
Ýlayer_metrics
¸	variables
¹trainable_variables
ºregularization_losses
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses*

Þtrace_0
ßtrace_1* 

àtrace_0
átrace_1* 
* 
* 
* 
* 

ânon_trainable_variables
ãlayers
ämetrics
 ålayer_regularization_losses
ælayer_metrics
¿	variables
Àtrainable_variables
Áregularization_losses
Ã__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses* 
¬
çtrace_0
ètrace_1
étrace_2
êtrace_3
ëtrace_4
ìtrace_5
ítrace_6
îtrace_7
ïtrace_8
ðtrace_9
ñtrace_10
òtrace_11* 
¬
ótrace_0
ôtrace_1
õtrace_2
ötrace_3
÷trace_4
øtrace_5
ùtrace_6
útrace_7
ûtrace_8
ütrace_9
ýtrace_10
þtrace_11* 
* 

F0
G1*

F0
G1*
* 

ÿnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Æ	variables
Çtrainable_variables
Èregularization_losses
Ê__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses*

trace_0* 

trace_0* 

H0
I1*

H0
I1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ì	variables
Ítrainable_variables
Îregularization_losses
Ð__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses*

trace_0* 

trace_0* 

J0
K1*

J0
K1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ò	variables
Ótrainable_variables
Ôregularization_losses
Ö__call__
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses*

trace_0* 

trace_0* 

L0
M1*

L0
M1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ø	variables
Ùtrainable_variables
Úregularization_losses
Ü__call__
+Ý&call_and_return_all_conditional_losses
'Ý"call_and_return_conditional_losses*

trace_0* 

trace_0* 

N0
O1*

N0
O1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Þ	variables
ßtrainable_variables
àregularization_losses
â__call__
+ã&call_and_return_all_conditional_losses
'ã"call_and_return_conditional_losses*

 trace_0* 

¡trace_0* 

P0
Q1*

P0
Q1*
* 

¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
ä	variables
åtrainable_variables
æregularization_losses
è__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses*

§trace_0* 

¨trace_0* 

R0
S1*

R0
S1*
* 

©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
ê	variables
ëtrainable_variables
ìregularization_losses
î__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses*

®trace_0* 

¯trace_0* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*

T0
U1*

ô	variables*

V0
W1*

ö	variables*

X0
Y1*

ø	variables*
* 
* 
* 
* 
* 
* 
* 
* 

&0
'1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

,0
-1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

20
31*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

80
91*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

>0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

D0
E1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

VARIABLE_VALUE+Adam/tic_tac_toe_neural_net/conv2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tic_tac_toe_neural_net/conv2d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE7Adam/tic_tac_toe_neural_net/batch_normalization/gamma/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE6Adam/tic_tac_toe_neural_net/batch_normalization/beta/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE-Adam/tic_tac_toe_neural_net/conv2d_1/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tic_tac_toe_neural_net/conv2d_1/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE9Adam/tic_tac_toe_neural_net/batch_normalization_1/gamma/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/tic_tac_toe_neural_net/batch_normalization_1/beta/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE-Adam/tic_tac_toe_neural_net/conv2d_2/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tic_tac_toe_neural_net/conv2d_2/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE9Adam/tic_tac_toe_neural_net/batch_normalization_2/gamma/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/tic_tac_toe_neural_net/batch_normalization_2/beta/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE-Adam/tic_tac_toe_neural_net/conv2d_3/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tic_tac_toe_neural_net/conv2d_3/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE9Adam/tic_tac_toe_neural_net/batch_normalization_3/gamma/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/tic_tac_toe_neural_net/batch_normalization_3/beta/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/tic_tac_toe_neural_net/dense/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE(Adam/tic_tac_toe_neural_net/dense/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE9Adam/tic_tac_toe_neural_net/batch_normalization_4/gamma/mCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/tic_tac_toe_neural_net/batch_normalization_4/beta/mCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/tic_tac_toe_neural_net/dense_1/kernel/mCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/tic_tac_toe_neural_net/dense_1/bias/mCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE9Adam/tic_tac_toe_neural_net/batch_normalization_5/gamma/mCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/tic_tac_toe_neural_net/batch_normalization_5/beta/mCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/tic_tac_toe_neural_net/dense_2/kernel/mCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/tic_tac_toe_neural_net/dense_2/bias/mCvariables/37/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/tic_tac_toe_neural_net/dense_3/kernel/mCvariables/38/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/tic_tac_toe_neural_net/dense_3/bias/mCvariables/39/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/tic_tac_toe_neural_net/dense_4/kernel/mCvariables/40/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/tic_tac_toe_neural_net/dense_4/bias/mCvariables/41/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/tic_tac_toe_neural_net/dense_5/kernel/mCvariables/42/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/tic_tac_toe_neural_net/dense_5/bias/mCvariables/43/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/tic_tac_toe_neural_net/dense_6/kernel/mCvariables/44/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/tic_tac_toe_neural_net/dense_6/bias/mCvariables/45/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/tic_tac_toe_neural_net/dense_7/kernel/mCvariables/46/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/tic_tac_toe_neural_net/dense_7/bias/mCvariables/47/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/tic_tac_toe_neural_net/dense_8/kernel/mCvariables/48/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/tic_tac_toe_neural_net/dense_8/bias/mCvariables/49/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tic_tac_toe_neural_net/conv2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/tic_tac_toe_neural_net/conv2d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE7Adam/tic_tac_toe_neural_net/batch_normalization/gamma/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE6Adam/tic_tac_toe_neural_net/batch_normalization/beta/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE-Adam/tic_tac_toe_neural_net/conv2d_1/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tic_tac_toe_neural_net/conv2d_1/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE9Adam/tic_tac_toe_neural_net/batch_normalization_1/gamma/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/tic_tac_toe_neural_net/batch_normalization_1/beta/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE-Adam/tic_tac_toe_neural_net/conv2d_2/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tic_tac_toe_neural_net/conv2d_2/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE9Adam/tic_tac_toe_neural_net/batch_normalization_2/gamma/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/tic_tac_toe_neural_net/batch_normalization_2/beta/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE-Adam/tic_tac_toe_neural_net/conv2d_3/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/tic_tac_toe_neural_net/conv2d_3/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE9Adam/tic_tac_toe_neural_net/batch_normalization_3/gamma/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/tic_tac_toe_neural_net/batch_normalization_3/beta/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/tic_tac_toe_neural_net/dense/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE(Adam/tic_tac_toe_neural_net/dense/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE9Adam/tic_tac_toe_neural_net/batch_normalization_4/gamma/vCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/tic_tac_toe_neural_net/batch_normalization_4/beta/vCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/tic_tac_toe_neural_net/dense_1/kernel/vCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/tic_tac_toe_neural_net/dense_1/bias/vCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE9Adam/tic_tac_toe_neural_net/batch_normalization_5/gamma/vCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE8Adam/tic_tac_toe_neural_net/batch_normalization_5/beta/vCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/tic_tac_toe_neural_net/dense_2/kernel/vCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/tic_tac_toe_neural_net/dense_2/bias/vCvariables/37/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/tic_tac_toe_neural_net/dense_3/kernel/vCvariables/38/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/tic_tac_toe_neural_net/dense_3/bias/vCvariables/39/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/tic_tac_toe_neural_net/dense_4/kernel/vCvariables/40/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/tic_tac_toe_neural_net/dense_4/bias/vCvariables/41/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/tic_tac_toe_neural_net/dense_5/kernel/vCvariables/42/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/tic_tac_toe_neural_net/dense_5/bias/vCvariables/43/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/tic_tac_toe_neural_net/dense_6/kernel/vCvariables/44/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/tic_tac_toe_neural_net/dense_6/bias/vCvariables/45/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/tic_tac_toe_neural_net/dense_7/kernel/vCvariables/46/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/tic_tac_toe_neural_net/dense_7/bias/vCvariables/47/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/tic_tac_toe_neural_net/dense_8/kernel/vCvariables/48/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/tic_tac_toe_neural_net/dense_8/bias/vCvariables/49/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_1Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
º
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1$tic_tac_toe_neural_net/conv2d/kernel"tic_tac_toe_neural_net/conv2d/bias0tic_tac_toe_neural_net/batch_normalization/gamma/tic_tac_toe_neural_net/batch_normalization/beta6tic_tac_toe_neural_net/batch_normalization/moving_mean:tic_tac_toe_neural_net/batch_normalization/moving_variance&tic_tac_toe_neural_net/conv2d_1/kernel$tic_tac_toe_neural_net/conv2d_1/bias2tic_tac_toe_neural_net/batch_normalization_1/gamma1tic_tac_toe_neural_net/batch_normalization_1/beta8tic_tac_toe_neural_net/batch_normalization_1/moving_mean<tic_tac_toe_neural_net/batch_normalization_1/moving_variance&tic_tac_toe_neural_net/conv2d_2/kernel$tic_tac_toe_neural_net/conv2d_2/bias2tic_tac_toe_neural_net/batch_normalization_2/gamma1tic_tac_toe_neural_net/batch_normalization_2/beta8tic_tac_toe_neural_net/batch_normalization_2/moving_mean<tic_tac_toe_neural_net/batch_normalization_2/moving_variance&tic_tac_toe_neural_net/conv2d_3/kernel$tic_tac_toe_neural_net/conv2d_3/bias2tic_tac_toe_neural_net/batch_normalization_3/gamma1tic_tac_toe_neural_net/batch_normalization_3/beta8tic_tac_toe_neural_net/batch_normalization_3/moving_mean<tic_tac_toe_neural_net/batch_normalization_3/moving_variance#tic_tac_toe_neural_net/dense/kernel!tic_tac_toe_neural_net/dense/bias<tic_tac_toe_neural_net/batch_normalization_4/moving_variance2tic_tac_toe_neural_net/batch_normalization_4/gamma8tic_tac_toe_neural_net/batch_normalization_4/moving_mean1tic_tac_toe_neural_net/batch_normalization_4/beta%tic_tac_toe_neural_net/dense_1/kernel#tic_tac_toe_neural_net/dense_1/bias<tic_tac_toe_neural_net/batch_normalization_5/moving_variance2tic_tac_toe_neural_net/batch_normalization_5/gamma8tic_tac_toe_neural_net/batch_normalization_5/moving_mean1tic_tac_toe_neural_net/batch_normalization_5/beta%tic_tac_toe_neural_net/dense_2/kernel#tic_tac_toe_neural_net/dense_2/bias%tic_tac_toe_neural_net/dense_3/kernel#tic_tac_toe_neural_net/dense_3/bias%tic_tac_toe_neural_net/dense_4/kernel#tic_tac_toe_neural_net/dense_4/bias%tic_tac_toe_neural_net/dense_5/kernel#tic_tac_toe_neural_net/dense_5/bias%tic_tac_toe_neural_net/dense_6/kernel#tic_tac_toe_neural_net/dense_6/bias%tic_tac_toe_neural_net/dense_7/kernel#tic_tac_toe_neural_net/dense_7/bias%tic_tac_toe_neural_net/dense_8/kernel#tic_tac_toe_neural_net/dense_8/bias*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ	:ÿÿÿÿÿÿÿÿÿ*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_41073775
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ÒH
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename8tic_tac_toe_neural_net/conv2d/kernel/Read/ReadVariableOp6tic_tac_toe_neural_net/conv2d/bias/Read/ReadVariableOpDtic_tac_toe_neural_net/batch_normalization/gamma/Read/ReadVariableOpCtic_tac_toe_neural_net/batch_normalization/beta/Read/ReadVariableOpJtic_tac_toe_neural_net/batch_normalization/moving_mean/Read/ReadVariableOpNtic_tac_toe_neural_net/batch_normalization/moving_variance/Read/ReadVariableOp:tic_tac_toe_neural_net/conv2d_1/kernel/Read/ReadVariableOp8tic_tac_toe_neural_net/conv2d_1/bias/Read/ReadVariableOpFtic_tac_toe_neural_net/batch_normalization_1/gamma/Read/ReadVariableOpEtic_tac_toe_neural_net/batch_normalization_1/beta/Read/ReadVariableOpLtic_tac_toe_neural_net/batch_normalization_1/moving_mean/Read/ReadVariableOpPtic_tac_toe_neural_net/batch_normalization_1/moving_variance/Read/ReadVariableOp:tic_tac_toe_neural_net/conv2d_2/kernel/Read/ReadVariableOp8tic_tac_toe_neural_net/conv2d_2/bias/Read/ReadVariableOpFtic_tac_toe_neural_net/batch_normalization_2/gamma/Read/ReadVariableOpEtic_tac_toe_neural_net/batch_normalization_2/beta/Read/ReadVariableOpLtic_tac_toe_neural_net/batch_normalization_2/moving_mean/Read/ReadVariableOpPtic_tac_toe_neural_net/batch_normalization_2/moving_variance/Read/ReadVariableOp:tic_tac_toe_neural_net/conv2d_3/kernel/Read/ReadVariableOp8tic_tac_toe_neural_net/conv2d_3/bias/Read/ReadVariableOpFtic_tac_toe_neural_net/batch_normalization_3/gamma/Read/ReadVariableOpEtic_tac_toe_neural_net/batch_normalization_3/beta/Read/ReadVariableOpLtic_tac_toe_neural_net/batch_normalization_3/moving_mean/Read/ReadVariableOpPtic_tac_toe_neural_net/batch_normalization_3/moving_variance/Read/ReadVariableOp7tic_tac_toe_neural_net/dense/kernel/Read/ReadVariableOp5tic_tac_toe_neural_net/dense/bias/Read/ReadVariableOpFtic_tac_toe_neural_net/batch_normalization_4/gamma/Read/ReadVariableOpEtic_tac_toe_neural_net/batch_normalization_4/beta/Read/ReadVariableOpLtic_tac_toe_neural_net/batch_normalization_4/moving_mean/Read/ReadVariableOpPtic_tac_toe_neural_net/batch_normalization_4/moving_variance/Read/ReadVariableOp9tic_tac_toe_neural_net/dense_1/kernel/Read/ReadVariableOp7tic_tac_toe_neural_net/dense_1/bias/Read/ReadVariableOpFtic_tac_toe_neural_net/batch_normalization_5/gamma/Read/ReadVariableOpEtic_tac_toe_neural_net/batch_normalization_5/beta/Read/ReadVariableOpLtic_tac_toe_neural_net/batch_normalization_5/moving_mean/Read/ReadVariableOpPtic_tac_toe_neural_net/batch_normalization_5/moving_variance/Read/ReadVariableOp9tic_tac_toe_neural_net/dense_2/kernel/Read/ReadVariableOp7tic_tac_toe_neural_net/dense_2/bias/Read/ReadVariableOp9tic_tac_toe_neural_net/dense_3/kernel/Read/ReadVariableOp7tic_tac_toe_neural_net/dense_3/bias/Read/ReadVariableOp9tic_tac_toe_neural_net/dense_4/kernel/Read/ReadVariableOp7tic_tac_toe_neural_net/dense_4/bias/Read/ReadVariableOp9tic_tac_toe_neural_net/dense_5/kernel/Read/ReadVariableOp7tic_tac_toe_neural_net/dense_5/bias/Read/ReadVariableOp9tic_tac_toe_neural_net/dense_6/kernel/Read/ReadVariableOp7tic_tac_toe_neural_net/dense_6/bias/Read/ReadVariableOp9tic_tac_toe_neural_net/dense_7/kernel/Read/ReadVariableOp7tic_tac_toe_neural_net/dense_7/bias/Read/ReadVariableOp9tic_tac_toe_neural_net/dense_8/kernel/Read/ReadVariableOp7tic_tac_toe_neural_net/dense_8/bias/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp?Adam/tic_tac_toe_neural_net/conv2d/kernel/m/Read/ReadVariableOp=Adam/tic_tac_toe_neural_net/conv2d/bias/m/Read/ReadVariableOpKAdam/tic_tac_toe_neural_net/batch_normalization/gamma/m/Read/ReadVariableOpJAdam/tic_tac_toe_neural_net/batch_normalization/beta/m/Read/ReadVariableOpAAdam/tic_tac_toe_neural_net/conv2d_1/kernel/m/Read/ReadVariableOp?Adam/tic_tac_toe_neural_net/conv2d_1/bias/m/Read/ReadVariableOpMAdam/tic_tac_toe_neural_net/batch_normalization_1/gamma/m/Read/ReadVariableOpLAdam/tic_tac_toe_neural_net/batch_normalization_1/beta/m/Read/ReadVariableOpAAdam/tic_tac_toe_neural_net/conv2d_2/kernel/m/Read/ReadVariableOp?Adam/tic_tac_toe_neural_net/conv2d_2/bias/m/Read/ReadVariableOpMAdam/tic_tac_toe_neural_net/batch_normalization_2/gamma/m/Read/ReadVariableOpLAdam/tic_tac_toe_neural_net/batch_normalization_2/beta/m/Read/ReadVariableOpAAdam/tic_tac_toe_neural_net/conv2d_3/kernel/m/Read/ReadVariableOp?Adam/tic_tac_toe_neural_net/conv2d_3/bias/m/Read/ReadVariableOpMAdam/tic_tac_toe_neural_net/batch_normalization_3/gamma/m/Read/ReadVariableOpLAdam/tic_tac_toe_neural_net/batch_normalization_3/beta/m/Read/ReadVariableOp>Adam/tic_tac_toe_neural_net/dense/kernel/m/Read/ReadVariableOp<Adam/tic_tac_toe_neural_net/dense/bias/m/Read/ReadVariableOpMAdam/tic_tac_toe_neural_net/batch_normalization_4/gamma/m/Read/ReadVariableOpLAdam/tic_tac_toe_neural_net/batch_normalization_4/beta/m/Read/ReadVariableOp@Adam/tic_tac_toe_neural_net/dense_1/kernel/m/Read/ReadVariableOp>Adam/tic_tac_toe_neural_net/dense_1/bias/m/Read/ReadVariableOpMAdam/tic_tac_toe_neural_net/batch_normalization_5/gamma/m/Read/ReadVariableOpLAdam/tic_tac_toe_neural_net/batch_normalization_5/beta/m/Read/ReadVariableOp@Adam/tic_tac_toe_neural_net/dense_2/kernel/m/Read/ReadVariableOp>Adam/tic_tac_toe_neural_net/dense_2/bias/m/Read/ReadVariableOp@Adam/tic_tac_toe_neural_net/dense_3/kernel/m/Read/ReadVariableOp>Adam/tic_tac_toe_neural_net/dense_3/bias/m/Read/ReadVariableOp@Adam/tic_tac_toe_neural_net/dense_4/kernel/m/Read/ReadVariableOp>Adam/tic_tac_toe_neural_net/dense_4/bias/m/Read/ReadVariableOp@Adam/tic_tac_toe_neural_net/dense_5/kernel/m/Read/ReadVariableOp>Adam/tic_tac_toe_neural_net/dense_5/bias/m/Read/ReadVariableOp@Adam/tic_tac_toe_neural_net/dense_6/kernel/m/Read/ReadVariableOp>Adam/tic_tac_toe_neural_net/dense_6/bias/m/Read/ReadVariableOp@Adam/tic_tac_toe_neural_net/dense_7/kernel/m/Read/ReadVariableOp>Adam/tic_tac_toe_neural_net/dense_7/bias/m/Read/ReadVariableOp@Adam/tic_tac_toe_neural_net/dense_8/kernel/m/Read/ReadVariableOp>Adam/tic_tac_toe_neural_net/dense_8/bias/m/Read/ReadVariableOp?Adam/tic_tac_toe_neural_net/conv2d/kernel/v/Read/ReadVariableOp=Adam/tic_tac_toe_neural_net/conv2d/bias/v/Read/ReadVariableOpKAdam/tic_tac_toe_neural_net/batch_normalization/gamma/v/Read/ReadVariableOpJAdam/tic_tac_toe_neural_net/batch_normalization/beta/v/Read/ReadVariableOpAAdam/tic_tac_toe_neural_net/conv2d_1/kernel/v/Read/ReadVariableOp?Adam/tic_tac_toe_neural_net/conv2d_1/bias/v/Read/ReadVariableOpMAdam/tic_tac_toe_neural_net/batch_normalization_1/gamma/v/Read/ReadVariableOpLAdam/tic_tac_toe_neural_net/batch_normalization_1/beta/v/Read/ReadVariableOpAAdam/tic_tac_toe_neural_net/conv2d_2/kernel/v/Read/ReadVariableOp?Adam/tic_tac_toe_neural_net/conv2d_2/bias/v/Read/ReadVariableOpMAdam/tic_tac_toe_neural_net/batch_normalization_2/gamma/v/Read/ReadVariableOpLAdam/tic_tac_toe_neural_net/batch_normalization_2/beta/v/Read/ReadVariableOpAAdam/tic_tac_toe_neural_net/conv2d_3/kernel/v/Read/ReadVariableOp?Adam/tic_tac_toe_neural_net/conv2d_3/bias/v/Read/ReadVariableOpMAdam/tic_tac_toe_neural_net/batch_normalization_3/gamma/v/Read/ReadVariableOpLAdam/tic_tac_toe_neural_net/batch_normalization_3/beta/v/Read/ReadVariableOp>Adam/tic_tac_toe_neural_net/dense/kernel/v/Read/ReadVariableOp<Adam/tic_tac_toe_neural_net/dense/bias/v/Read/ReadVariableOpMAdam/tic_tac_toe_neural_net/batch_normalization_4/gamma/v/Read/ReadVariableOpLAdam/tic_tac_toe_neural_net/batch_normalization_4/beta/v/Read/ReadVariableOp@Adam/tic_tac_toe_neural_net/dense_1/kernel/v/Read/ReadVariableOp>Adam/tic_tac_toe_neural_net/dense_1/bias/v/Read/ReadVariableOpMAdam/tic_tac_toe_neural_net/batch_normalization_5/gamma/v/Read/ReadVariableOpLAdam/tic_tac_toe_neural_net/batch_normalization_5/beta/v/Read/ReadVariableOp@Adam/tic_tac_toe_neural_net/dense_2/kernel/v/Read/ReadVariableOp>Adam/tic_tac_toe_neural_net/dense_2/bias/v/Read/ReadVariableOp@Adam/tic_tac_toe_neural_net/dense_3/kernel/v/Read/ReadVariableOp>Adam/tic_tac_toe_neural_net/dense_3/bias/v/Read/ReadVariableOp@Adam/tic_tac_toe_neural_net/dense_4/kernel/v/Read/ReadVariableOp>Adam/tic_tac_toe_neural_net/dense_4/bias/v/Read/ReadVariableOp@Adam/tic_tac_toe_neural_net/dense_5/kernel/v/Read/ReadVariableOp>Adam/tic_tac_toe_neural_net/dense_5/bias/v/Read/ReadVariableOp@Adam/tic_tac_toe_neural_net/dense_6/kernel/v/Read/ReadVariableOp>Adam/tic_tac_toe_neural_net/dense_6/bias/v/Read/ReadVariableOp@Adam/tic_tac_toe_neural_net/dense_7/kernel/v/Read/ReadVariableOp>Adam/tic_tac_toe_neural_net/dense_7/bias/v/Read/ReadVariableOp@Adam/tic_tac_toe_neural_net/dense_8/kernel/v/Read/ReadVariableOp>Adam/tic_tac_toe_neural_net/dense_8/bias/v/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_save_41075719
­3
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename$tic_tac_toe_neural_net/conv2d/kernel"tic_tac_toe_neural_net/conv2d/bias0tic_tac_toe_neural_net/batch_normalization/gamma/tic_tac_toe_neural_net/batch_normalization/beta6tic_tac_toe_neural_net/batch_normalization/moving_mean:tic_tac_toe_neural_net/batch_normalization/moving_variance&tic_tac_toe_neural_net/conv2d_1/kernel$tic_tac_toe_neural_net/conv2d_1/bias2tic_tac_toe_neural_net/batch_normalization_1/gamma1tic_tac_toe_neural_net/batch_normalization_1/beta8tic_tac_toe_neural_net/batch_normalization_1/moving_mean<tic_tac_toe_neural_net/batch_normalization_1/moving_variance&tic_tac_toe_neural_net/conv2d_2/kernel$tic_tac_toe_neural_net/conv2d_2/bias2tic_tac_toe_neural_net/batch_normalization_2/gamma1tic_tac_toe_neural_net/batch_normalization_2/beta8tic_tac_toe_neural_net/batch_normalization_2/moving_mean<tic_tac_toe_neural_net/batch_normalization_2/moving_variance&tic_tac_toe_neural_net/conv2d_3/kernel$tic_tac_toe_neural_net/conv2d_3/bias2tic_tac_toe_neural_net/batch_normalization_3/gamma1tic_tac_toe_neural_net/batch_normalization_3/beta8tic_tac_toe_neural_net/batch_normalization_3/moving_mean<tic_tac_toe_neural_net/batch_normalization_3/moving_variance#tic_tac_toe_neural_net/dense/kernel!tic_tac_toe_neural_net/dense/bias2tic_tac_toe_neural_net/batch_normalization_4/gamma1tic_tac_toe_neural_net/batch_normalization_4/beta8tic_tac_toe_neural_net/batch_normalization_4/moving_mean<tic_tac_toe_neural_net/batch_normalization_4/moving_variance%tic_tac_toe_neural_net/dense_1/kernel#tic_tac_toe_neural_net/dense_1/bias2tic_tac_toe_neural_net/batch_normalization_5/gamma1tic_tac_toe_neural_net/batch_normalization_5/beta8tic_tac_toe_neural_net/batch_normalization_5/moving_mean<tic_tac_toe_neural_net/batch_normalization_5/moving_variance%tic_tac_toe_neural_net/dense_2/kernel#tic_tac_toe_neural_net/dense_2/bias%tic_tac_toe_neural_net/dense_3/kernel#tic_tac_toe_neural_net/dense_3/bias%tic_tac_toe_neural_net/dense_4/kernel#tic_tac_toe_neural_net/dense_4/bias%tic_tac_toe_neural_net/dense_5/kernel#tic_tac_toe_neural_net/dense_5/bias%tic_tac_toe_neural_net/dense_6/kernel#tic_tac_toe_neural_net/dense_6/bias%tic_tac_toe_neural_net/dense_7/kernel#tic_tac_toe_neural_net/dense_7/bias%tic_tac_toe_neural_net/dense_8/kernel#tic_tac_toe_neural_net/dense_8/biastotal_2count_2total_1count_1totalcount	Adam/iterAdam/beta_1Adam/beta_2
Adam/decay+Adam/tic_tac_toe_neural_net/conv2d/kernel/m)Adam/tic_tac_toe_neural_net/conv2d/bias/m7Adam/tic_tac_toe_neural_net/batch_normalization/gamma/m6Adam/tic_tac_toe_neural_net/batch_normalization/beta/m-Adam/tic_tac_toe_neural_net/conv2d_1/kernel/m+Adam/tic_tac_toe_neural_net/conv2d_1/bias/m9Adam/tic_tac_toe_neural_net/batch_normalization_1/gamma/m8Adam/tic_tac_toe_neural_net/batch_normalization_1/beta/m-Adam/tic_tac_toe_neural_net/conv2d_2/kernel/m+Adam/tic_tac_toe_neural_net/conv2d_2/bias/m9Adam/tic_tac_toe_neural_net/batch_normalization_2/gamma/m8Adam/tic_tac_toe_neural_net/batch_normalization_2/beta/m-Adam/tic_tac_toe_neural_net/conv2d_3/kernel/m+Adam/tic_tac_toe_neural_net/conv2d_3/bias/m9Adam/tic_tac_toe_neural_net/batch_normalization_3/gamma/m8Adam/tic_tac_toe_neural_net/batch_normalization_3/beta/m*Adam/tic_tac_toe_neural_net/dense/kernel/m(Adam/tic_tac_toe_neural_net/dense/bias/m9Adam/tic_tac_toe_neural_net/batch_normalization_4/gamma/m8Adam/tic_tac_toe_neural_net/batch_normalization_4/beta/m,Adam/tic_tac_toe_neural_net/dense_1/kernel/m*Adam/tic_tac_toe_neural_net/dense_1/bias/m9Adam/tic_tac_toe_neural_net/batch_normalization_5/gamma/m8Adam/tic_tac_toe_neural_net/batch_normalization_5/beta/m,Adam/tic_tac_toe_neural_net/dense_2/kernel/m*Adam/tic_tac_toe_neural_net/dense_2/bias/m,Adam/tic_tac_toe_neural_net/dense_3/kernel/m*Adam/tic_tac_toe_neural_net/dense_3/bias/m,Adam/tic_tac_toe_neural_net/dense_4/kernel/m*Adam/tic_tac_toe_neural_net/dense_4/bias/m,Adam/tic_tac_toe_neural_net/dense_5/kernel/m*Adam/tic_tac_toe_neural_net/dense_5/bias/m,Adam/tic_tac_toe_neural_net/dense_6/kernel/m*Adam/tic_tac_toe_neural_net/dense_6/bias/m,Adam/tic_tac_toe_neural_net/dense_7/kernel/m*Adam/tic_tac_toe_neural_net/dense_7/bias/m,Adam/tic_tac_toe_neural_net/dense_8/kernel/m*Adam/tic_tac_toe_neural_net/dense_8/bias/m+Adam/tic_tac_toe_neural_net/conv2d/kernel/v)Adam/tic_tac_toe_neural_net/conv2d/bias/v7Adam/tic_tac_toe_neural_net/batch_normalization/gamma/v6Adam/tic_tac_toe_neural_net/batch_normalization/beta/v-Adam/tic_tac_toe_neural_net/conv2d_1/kernel/v+Adam/tic_tac_toe_neural_net/conv2d_1/bias/v9Adam/tic_tac_toe_neural_net/batch_normalization_1/gamma/v8Adam/tic_tac_toe_neural_net/batch_normalization_1/beta/v-Adam/tic_tac_toe_neural_net/conv2d_2/kernel/v+Adam/tic_tac_toe_neural_net/conv2d_2/bias/v9Adam/tic_tac_toe_neural_net/batch_normalization_2/gamma/v8Adam/tic_tac_toe_neural_net/batch_normalization_2/beta/v-Adam/tic_tac_toe_neural_net/conv2d_3/kernel/v+Adam/tic_tac_toe_neural_net/conv2d_3/bias/v9Adam/tic_tac_toe_neural_net/batch_normalization_3/gamma/v8Adam/tic_tac_toe_neural_net/batch_normalization_3/beta/v*Adam/tic_tac_toe_neural_net/dense/kernel/v(Adam/tic_tac_toe_neural_net/dense/bias/v9Adam/tic_tac_toe_neural_net/batch_normalization_4/gamma/v8Adam/tic_tac_toe_neural_net/batch_normalization_4/beta/v,Adam/tic_tac_toe_neural_net/dense_1/kernel/v*Adam/tic_tac_toe_neural_net/dense_1/bias/v9Adam/tic_tac_toe_neural_net/batch_normalization_5/gamma/v8Adam/tic_tac_toe_neural_net/batch_normalization_5/beta/v,Adam/tic_tac_toe_neural_net/dense_2/kernel/v*Adam/tic_tac_toe_neural_net/dense_2/bias/v,Adam/tic_tac_toe_neural_net/dense_3/kernel/v*Adam/tic_tac_toe_neural_net/dense_3/bias/v,Adam/tic_tac_toe_neural_net/dense_4/kernel/v*Adam/tic_tac_toe_neural_net/dense_4/bias/v,Adam/tic_tac_toe_neural_net/dense_5/kernel/v*Adam/tic_tac_toe_neural_net/dense_5/bias/v,Adam/tic_tac_toe_neural_net/dense_6/kernel/v*Adam/tic_tac_toe_neural_net/dense_6/bias/v,Adam/tic_tac_toe_neural_net/dense_7/kernel/v*Adam/tic_tac_toe_neural_net/dense_7/bias/v,Adam/tic_tac_toe_neural_net/dense_8/kernel/v*Adam/tic_tac_toe_neural_net/dense_8/bias/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__traced_restore_41076137¢
	
×
8__inference_batch_normalization_1_layer_call_fn_41074578

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_41071924
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë

*__inference_dense_1_layer_call_fn_41074895

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_41072389p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

F
*__inference_dropout_layer_call_fn_41075000

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072432`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
®

ÿ
D__inference_conv2d_layer_call_and_return_conditional_losses_41074471

inputs9
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç

*__inference_dense_6_layer_call_fn_41075236

inputs
unknown:	2
	unknown_0:2
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_41072508o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_41071829

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ä
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_41071860

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
¢
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_41071957

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
×
8__inference_batch_normalization_4_layer_call_fn_41074819

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_41072087p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
¢
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_41074596

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
c
E__inference_dropout_layer_call_and_return_conditional_losses_41075065

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±%
ð
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_41072216

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦ñ
.
T__inference_tic_tac_toe_neural_net_layer_call_and_return_conditional_losses_41074452
x@
%conv2d_conv2d_readvariableop_resource:5
&conv2d_biasadd_readvariableop_resource:	:
+batch_normalization_readvariableop_resource:	<
-batch_normalization_readvariableop_1_resource:	K
<batch_normalization_fusedbatchnormv3_readvariableop_resource:	M
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_1_conv2d_readvariableop_resource:7
(conv2d_1_biasadd_readvariableop_resource:	<
-batch_normalization_1_readvariableop_resource:	>
/batch_normalization_1_readvariableop_1_resource:	M
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_2_conv2d_readvariableop_resource:7
(conv2d_2_biasadd_readvariableop_resource:	<
-batch_normalization_2_readvariableop_resource:	>
/batch_normalization_2_readvariableop_1_resource:	M
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_3_conv2d_readvariableop_resource:7
(conv2d_3_biasadd_readvariableop_resource:	<
-batch_normalization_3_readvariableop_resource:	>
/batch_normalization_3_readvariableop_1_resource:	M
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	8
$dense_matmul_readvariableop_resource:
4
%dense_biasadd_readvariableop_resource:	L
=batch_normalization_4_assignmovingavg_readvariableop_resource:	N
?batch_normalization_4_assignmovingavg_1_readvariableop_resource:	J
;batch_normalization_4_batchnorm_mul_readvariableop_resource:	F
7batch_normalization_4_batchnorm_readvariableop_resource:	:
&dense_1_matmul_readvariableop_resource:
6
'dense_1_biasadd_readvariableop_resource:	L
=batch_normalization_5_assignmovingavg_readvariableop_resource:	N
?batch_normalization_5_assignmovingavg_1_readvariableop_resource:	J
;batch_normalization_5_batchnorm_mul_readvariableop_resource:	F
7batch_normalization_5_batchnorm_readvariableop_resource:	9
&dense_2_matmul_readvariableop_resource:	25
'dense_2_biasadd_readvariableop_resource:28
&dense_3_matmul_readvariableop_resource:25
'dense_3_biasadd_readvariableop_resource:8
&dense_4_matmul_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:8
&dense_5_matmul_readvariableop_resource:	5
'dense_5_biasadd_readvariableop_resource:	9
&dense_6_matmul_readvariableop_resource:	25
'dense_6_biasadd_readvariableop_resource:28
&dense_7_matmul_readvariableop_resource:2(5
'dense_7_biasadd_readvariableop_resource:(8
&dense_8_matmul_readvariableop_resource:(5
'dense_8_biasadd_readvariableop_resource:
identity

identity_1¢"batch_normalization/AssignNewValue¢$batch_normalization/AssignNewValue_1¢3batch_normalization/FusedBatchNormV3/ReadVariableOp¢5batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢"batch_normalization/ReadVariableOp¢$batch_normalization/ReadVariableOp_1¢$batch_normalization_1/AssignNewValue¢&batch_normalization_1/AssignNewValue_1¢5batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_1/ReadVariableOp¢&batch_normalization_1/ReadVariableOp_1¢$batch_normalization_2/AssignNewValue¢&batch_normalization_2/AssignNewValue_1¢5batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_2/ReadVariableOp¢&batch_normalization_2/ReadVariableOp_1¢$batch_normalization_3/AssignNewValue¢&batch_normalization_3/AssignNewValue_1¢5batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_3/ReadVariableOp¢&batch_normalization_3/ReadVariableOp_1¢%batch_normalization_4/AssignMovingAvg¢4batch_normalization_4/AssignMovingAvg/ReadVariableOp¢'batch_normalization_4/AssignMovingAvg_1¢6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_4/batchnorm/ReadVariableOp¢2batch_normalization_4/batchnorm/mul/ReadVariableOp¢%batch_normalization_5/AssignMovingAvg¢4batch_normalization_5/AssignMovingAvg/ReadVariableOp¢'batch_normalization_5/AssignMovingAvg_1¢6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_5/batchnorm/ReadVariableOp¢2batch_normalization_5/batchnorm/mul/ReadVariableOp¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp¢conv2d_3/BiasAdd/ReadVariableOp¢conv2d_3/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢dense_8/MatMul/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0£
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes	
:*
dtype0
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes	
:*
dtype0­
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0±
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0¾
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape( 
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(q
ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¸
conv2d_1/Conv2DConv2DRelu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ê
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¨
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(u
Relu_1Relu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0º
conv2d_2/Conv2DConv2DRelu_1:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ê
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¨
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(u
Relu_2Relu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0»
conv2d_3/Conv2DConv2DRelu_2:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ê
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¨
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(u
Relu_3Relu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   {
flatten/ReshapeReshapeRelu_3:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¼
"batch_normalization_4/moments/meanMeandense/BiasAdd:output:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:	Ä
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencedense/BiasAdd:output:03batch_normalization_4/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: á
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
  
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 p
+batch_normalization_4/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¯
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0Ä
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*
_output_shapes	
:»
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:
%batch_normalization_4/AssignMovingAvgAssignSubVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<³
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0Ê
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Á
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:
'batch_normalization_4/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:´
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:}
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:«
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0·
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
: 
%batch_normalization_4/batchnorm/mul_1Muldense/BiasAdd:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:£
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0³
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:µ
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
Relu_4Relu)batch_normalization_4/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout/dropout/MulMulRelu_4:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
dropout/dropout/ShapeShapeRelu_4:activations:0*
T0*
_output_shapes
:
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¿
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
4batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¾
"batch_normalization_5/moments/meanMeandense_1/BiasAdd:output:0=batch_normalization_5/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(
*batch_normalization_5/moments/StopGradientStopGradient+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
:	Æ
/batch_normalization_5/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_5/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8batch_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: á
&batch_normalization_5/moments/varianceMean3batch_normalization_5/moments/SquaredDifference:z:0Abatch_normalization_5/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(
%batch_normalization_5/moments/SqueezeSqueeze+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
  
'batch_normalization_5/moments/Squeeze_1Squeeze/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 p
+batch_normalization_5/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¯
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_5_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0Ä
)batch_normalization_5/AssignMovingAvg/subSub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_5/moments/Squeeze:output:0*
T0*
_output_shapes	
:»
)batch_normalization_5/AssignMovingAvg/mulMul-batch_normalization_5/AssignMovingAvg/sub:z:04batch_normalization_5/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:
%batch_normalization_5/AssignMovingAvgAssignSubVariableOp=batch_normalization_5_assignmovingavg_readvariableop_resource-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_5/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<³
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_5_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0Ê
+batch_normalization_5/AssignMovingAvg_1/subSub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_5/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Á
+batch_normalization_5/AssignMovingAvg_1/mulMul/batch_normalization_5/AssignMovingAvg_1/sub:z:06batch_normalization_5/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:
'batch_normalization_5/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_5_assignmovingavg_1_readvariableop_resource/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:´
#batch_normalization_5/batchnorm/addAddV20batch_normalization_5/moments/Squeeze_1:output:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes	
:}
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes	
:«
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0·
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:¢
%batch_normalization_5/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
%batch_normalization_5/batchnorm/mul_2Mul.batch_normalization_5/moments/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes	
:£
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0³
#batch_normalization_5/batchnorm/subSub6batch_normalization_5/batchnorm/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:µ
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
Relu_5Relu)batch_normalization_5/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout/dropout_1/MulMulRelu_5:activations:0 dropout/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
dropout/dropout_1/ShapeShapeRelu_5:activations:0*
T0*
_output_shapes
:¡
.dropout/dropout_1/random_uniform/RandomUniformRandomUniform dropout/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Å
dropout/dropout_1/GreaterEqualGreaterEqual7dropout/dropout_1/random_uniform/RandomUniform:output:0)dropout/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout_1/CastCast"dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout_1/Mul_1Muldropout/dropout_1/Mul:z:0dropout/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype0
dense_2/MatMulMatMuldropout/dropout_1/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2\
dropout/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout/dropout_2/MulMuldense_2/Relu:activations:0 dropout/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2a
dropout/dropout_2/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
: 
.dropout/dropout_2/random_uniform/RandomUniformRandomUniform dropout/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
dtype0e
 dropout/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ä
dropout/dropout_2/GreaterEqualGreaterEqual7dropout/dropout_2/random_uniform/RandomUniform:output:0)dropout/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/dropout_2/CastCast"dropout/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/dropout_2/Mul_1Muldropout/dropout_2/Mul:z:0dropout/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
dense_3/MatMulMatMuldropout/dropout_2/Mul_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout/dropout_3/MulMuldense_3/Relu:activations:0 dropout/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
dropout/dropout_3/ShapeShapedense_3/Relu:activations:0*
T0*
_output_shapes
: 
.dropout/dropout_3/random_uniform/RandomUniformRandomUniform dropout/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ä
dropout/dropout_3/GreaterEqualGreaterEqual7dropout/dropout_3/random_uniform/RandomUniform:output:0)dropout/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout_3/CastCast"dropout/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout_3/Mul_1Muldropout/dropout_3/Mul:z:0dropout/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_4/MatMulMatMuldropout/dropout_3/Mul_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout/dropout_4/MulMuldense_4/Relu:activations:0 dropout/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
dropout/dropout_4/ShapeShapedense_4/Relu:activations:0*
T0*
_output_shapes
: 
.dropout/dropout_4/random_uniform/RandomUniformRandomUniform dropout/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ä
dropout/dropout_4/GreaterEqualGreaterEqual7dropout/dropout_4/random_uniform/RandomUniform:output:0)dropout/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout_4/CastCast"dropout/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout_4/Mul_1Muldropout/dropout_4/Mul:z:0dropout/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:	*
dtype0
dense_5/MatMulMatMuldropout/dropout_4/Mul_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	f
dense_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype0
dense_6/MatMulMatMuldropout/dropout_1/Mul_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2\
dropout/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout/dropout_5/MulMuldense_6/Relu:activations:0 dropout/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2a
dropout/dropout_5/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
: 
.dropout/dropout_5/random_uniform/RandomUniformRandomUniform dropout/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
dtype0e
 dropout/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ä
dropout/dropout_5/GreaterEqualGreaterEqual7dropout/dropout_5/random_uniform/RandomUniform:output:0)dropout/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/dropout_5/CastCast"dropout/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/dropout_5/Mul_1Muldropout/dropout_5/Mul:z:0dropout/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype0
dense_7/MatMulMatMuldropout/dropout_5/Mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
dropout/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout/dropout_6/MulMuldense_7/Relu:activations:0 dropout/dropout_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(a
dropout/dropout_6/ShapeShapedense_7/Relu:activations:0*
T0*
_output_shapes
: 
.dropout/dropout_6/random_uniform/RandomUniformRandomUniform dropout/dropout_6/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
dtype0e
 dropout/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ä
dropout/dropout_6/GreaterEqualGreaterEqual7dropout/dropout_6/random_uniform/RandomUniform:output:0)dropout/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
dropout/dropout_6/CastCast"dropout/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
dropout/dropout_6/Mul_1Muldropout/dropout_6/Mul:z:0dropout/dropout_6/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0
dense_8/MatMulMatMuldropout/dropout_6/Mul_1:z:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_8/TanhTanhdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_5/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	a

Identity_1Identitydense_8/Tanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1&^batch_normalization_4/AssignMovingAvg5^batch_normalization_4/AssignMovingAvg/ReadVariableOp(^batch_normalization_4/AssignMovingAvg_17^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp3^batch_normalization_4/batchnorm/mul/ReadVariableOp&^batch_normalization_5/AssignMovingAvg5^batch_normalization_5/AssignMovingAvg/ReadVariableOp(^batch_normalization_5/AssignMovingAvg_17^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp3^batch_normalization_5/batchnorm/mul/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12N
%batch_normalization_4/AssignMovingAvg%batch_normalization_4/AssignMovingAvg2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_4/AssignMovingAvg_1'batch_normalization_4/AssignMovingAvg_12p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp2N
%batch_normalization_5/AssignMovingAvg%batch_normalization_5/AssignMovingAvg2l
4batch_normalization_5/AssignMovingAvg/ReadVariableOp4batch_normalization_5/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_5/AssignMovingAvg_1'batch_normalization_5/AssignMovingAvg_12p
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_5/batchnorm/ReadVariableOp.batch_normalization_5/batchnorm/ReadVariableOp2h
2batch_normalization_5/batchnorm/mul/ReadVariableOp2batch_normalization_5/batchnorm/mul/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
¡

ö
E__inference_dense_5_layer_call_and_return_conditional_losses_41075227

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶
ö
&__inference_signature_wrapper_41073775
input_1"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:


unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:


unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	

unknown_34:	

unknown_35:	2

unknown_36:2

unknown_37:2

unknown_38:

unknown_39:

unknown_40:

unknown_41:	

unknown_42:	

unknown_43:	2

unknown_44:2

unknown_45:2(

unknown_46:(

unknown_47:(

unknown_48:
identity

identity_1¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ	:ÿÿÿÿÿÿÿÿÿ*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_41071806o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ñ	
÷
C__inference_dense_layer_call_and_return_conditional_losses_41074806

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
¶
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_41072087

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
¢
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_41074677

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ	
÷
C__inference_dense_layer_call_and_return_conditional_losses_41072356

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
×
8__inference_batch_normalization_2_layer_call_fn_41074659

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_41071988
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Æ
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_41071988

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
a
E__inference_flatten_layer_call_and_return_conditional_losses_41074787

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û	
d
E__inference_dropout_layer_call_and_return_conditional_losses_41075135

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
×
8__inference_batch_normalization_5_layer_call_fn_41074931

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_41072216p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó	
d
E__inference_dropout_layer_call_and_return_conditional_losses_41075099

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
Ä

*__inference_dense_5_layer_call_fn_41075216

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_41072491o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç

*__inference_dense_2_layer_call_fn_41075156

inputs
unknown:	2
	unknown_0:2
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_41072422o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
F
*__inference_flatten_layer_call_fn_41074781

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_41072344a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
¢
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_41072021

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
×
8__inference_batch_normalization_1_layer_call_fn_41074565

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_41071893
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
c
E__inference_dropout_layer_call_and_return_conditional_losses_41072409

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±%
ð
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_41074886

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
¢
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_41074758

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
c
*__inference_dropout_layer_call_fn_41074995

inputs
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072692o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
û	
d
E__inference_dropout_layer_call_and_return_conditional_losses_41072840

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
×
T__inference_tic_tac_toe_neural_net_layer_call_and_return_conditional_losses_41073178
x*
conv2d_41073043:
conv2d_41073045:	+
batch_normalization_41073048:	+
batch_normalization_41073050:	+
batch_normalization_41073052:	+
batch_normalization_41073054:	-
conv2d_1_41073058: 
conv2d_1_41073060:	-
batch_normalization_1_41073063:	-
batch_normalization_1_41073065:	-
batch_normalization_1_41073067:	-
batch_normalization_1_41073069:	-
conv2d_2_41073073: 
conv2d_2_41073075:	-
batch_normalization_2_41073078:	-
batch_normalization_2_41073080:	-
batch_normalization_2_41073082:	-
batch_normalization_2_41073084:	-
conv2d_3_41073088: 
conv2d_3_41073090:	-
batch_normalization_3_41073093:	-
batch_normalization_3_41073095:	-
batch_normalization_3_41073097:	-
batch_normalization_3_41073099:	"
dense_41073104:

dense_41073106:	-
batch_normalization_4_41073109:	-
batch_normalization_4_41073111:	-
batch_normalization_4_41073113:	-
batch_normalization_4_41073115:	$
dense_1_41073120:

dense_1_41073122:	-
batch_normalization_5_41073125:	-
batch_normalization_5_41073127:	-
batch_normalization_5_41073129:	-
batch_normalization_5_41073131:	#
dense_2_41073136:	2
dense_2_41073138:2"
dense_3_41073142:2
dense_3_41073144:"
dense_4_41073148:
dense_4_41073150:"
dense_5_41073154:	
dense_5_41073156:	#
dense_6_41073159:	2
dense_6_41073161:2"
dense_7_41073165:2(
dense_7_41073167:("
dense_8_41073171:(
dense_8_41073173:
identity

identity_1¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout/StatefulPartitionedCall_1¢!dropout/StatefulPartitionedCall_2¢!dropout/StatefulPartitionedCall_3¢!dropout/StatefulPartitionedCall_4¢!dropout/StatefulPartitionedCall_5¢!dropout/StatefulPartitionedCall_6ò
conv2d/StatefulPartitionedCallStatefulPartitionedCallxconv2d_41073043conv2d_41073045*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_41072244
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_41073048batch_normalization_41073050batch_normalization_41073052batch_normalization_41073054*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_41071860}
ReluRelu4batch_normalization/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv2d_1_41073058conv2d_1_41073060*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_41072270
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_41073063batch_normalization_1_41073065batch_normalization_1_41073067batch_normalization_1_41073069*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_41071924
Relu_1Relu6batch_normalization_1/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0conv2d_2_41073073conv2d_2_41073075*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_41072296
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_41073078batch_normalization_2_41073080batch_normalization_2_41073082batch_normalization_2_41073084*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_41071988
Relu_2Relu6batch_normalization_2/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0conv2d_3_41073088conv2d_3_41073090*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_41072322
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_41073093batch_normalization_3_41073095batch_normalization_3_41073097batch_normalization_3_41073099*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_41072052
Relu_3Relu6batch_normalization_3/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
flatten/PartitionedCallPartitionedCallRelu_3:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_41072344
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_41073104dense_41073106*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_41072356
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_4_41073109batch_normalization_4_41073111batch_normalization_4_41073113batch_normalization_4_41073115*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_41072134y
Relu_4Relu6batch_normalization_4/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
dropout/StatefulPartitionedCallStatefulPartitionedCallRelu_4:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072872
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_41073120dense_1_41073122*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_41072389
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_5_41073125batch_normalization_5_41073127batch_normalization_5_41073129batch_normalization_5_41073131*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_41072216y
Relu_5Relu6batch_normalization_5/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿû
!dropout/StatefulPartitionedCall_1StatefulPartitionedCallRelu_5:activations:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072840
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout/StatefulPartitionedCall_1:output:0dense_2_41073136dense_2_41073138*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_41072422
!dropout/StatefulPartitionedCall_2StatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0"^dropout/StatefulPartitionedCall_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072724
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout/StatefulPartitionedCall_2:output:0dense_3_41073142dense_3_41073144*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_41072445
!dropout/StatefulPartitionedCall_3StatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0"^dropout/StatefulPartitionedCall_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072798
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout/StatefulPartitionedCall_3:output:0dense_4_41073148dense_4_41073150*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_41072468
!dropout/StatefulPartitionedCall_4StatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0"^dropout/StatefulPartitionedCall_3*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072766
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout/StatefulPartitionedCall_4:output:0dense_5_41073154dense_5_41073156*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_41072491
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout/StatefulPartitionedCall_1:output:0dense_6_41073159dense_6_41073161*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_41072508
!dropout/StatefulPartitionedCall_5StatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0"^dropout/StatefulPartitionedCall_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072724
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout/StatefulPartitionedCall_5:output:0dense_7_41073165dense_7_41073167*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_41072526
!dropout/StatefulPartitionedCall_6StatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0"^dropout/StatefulPartitionedCall_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072692
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout/StatefulPartitionedCall_6:output:0dense_8_41073171dense_8_41073173*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_41072549w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	y

Identity_1Identity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout/StatefulPartitionedCall_1"^dropout/StatefulPartitionedCall_2"^dropout/StatefulPartitionedCall_3"^dropout/StatefulPartitionedCall_4"^dropout/StatefulPartitionedCall_5"^dropout/StatefulPartitionedCall_6*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout/StatefulPartitionedCall_1!dropout/StatefulPartitionedCall_12F
!dropout/StatefulPartitionedCall_2!dropout/StatefulPartitionedCall_22F
!dropout/StatefulPartitionedCall_3!dropout/StatefulPartitionedCall_32F
!dropout/StatefulPartitionedCall_4!dropout/StatefulPartitionedCall_42F
!dropout/StatefulPartitionedCall_5!dropout/StatefulPartitionedCall_52F
!dropout/StatefulPartitionedCall_6!dropout/StatefulPartitionedCall_6:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex


ö
E__inference_dense_7_layer_call_and_return_conditional_losses_41075267

inputs0
matmul_readvariableop_resource:2(-
biasadd_readvariableop_resource:(
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs


ö
E__inference_dense_8_layer_call_and_return_conditional_losses_41072549

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
Ü

9__inference_tic_tac_toe_neural_net_layer_call_fn_41073989
x"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:


unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:


unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	

unknown_34:	

unknown_35:	2

unknown_36:2

unknown_37:2

unknown_38:

unknown_39:

unknown_40:

unknown_41:	

unknown_42:	

unknown_43:	2

unknown_44:2

unknown_45:2(

unknown_46:(

unknown_47:(

unknown_48:
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ	:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 #$%&'()*+,-./012*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_tic_tac_toe_neural_net_layer_call_and_return_conditional_losses_41073178o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
õ
£
+__inference_conv2d_2_layer_call_fn_41074623

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_41072296x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
c
*__inference_dropout_layer_call_fn_41075005

inputs
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072724o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ222
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
ó	
d
E__inference_dropout_layer_call_and_return_conditional_losses_41072692

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
±%
ð
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_41074985

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

*__inference_dense_8_layer_call_fn_41075276

inputs
unknown:(
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_41072549o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
õ
Ý
T__inference_tic_tac_toe_neural_net_layer_call_and_return_conditional_losses_41072557
x*
conv2d_41072245:
conv2d_41072247:	+
batch_normalization_41072250:	+
batch_normalization_41072252:	+
batch_normalization_41072254:	+
batch_normalization_41072256:	-
conv2d_1_41072271: 
conv2d_1_41072273:	-
batch_normalization_1_41072276:	-
batch_normalization_1_41072278:	-
batch_normalization_1_41072280:	-
batch_normalization_1_41072282:	-
conv2d_2_41072297: 
conv2d_2_41072299:	-
batch_normalization_2_41072302:	-
batch_normalization_2_41072304:	-
batch_normalization_2_41072306:	-
batch_normalization_2_41072308:	-
conv2d_3_41072323: 
conv2d_3_41072325:	-
batch_normalization_3_41072328:	-
batch_normalization_3_41072330:	-
batch_normalization_3_41072332:	-
batch_normalization_3_41072334:	"
dense_41072357:

dense_41072359:	-
batch_normalization_4_41072362:	-
batch_normalization_4_41072364:	-
batch_normalization_4_41072366:	-
batch_normalization_4_41072368:	$
dense_1_41072390:

dense_1_41072392:	-
batch_normalization_5_41072395:	-
batch_normalization_5_41072397:	-
batch_normalization_5_41072399:	-
batch_normalization_5_41072401:	#
dense_2_41072423:	2
dense_2_41072425:2"
dense_3_41072446:2
dense_3_41072448:"
dense_4_41072469:
dense_4_41072471:"
dense_5_41072492:	
dense_5_41072494:	#
dense_6_41072509:	2
dense_6_41072511:2"
dense_7_41072527:2(
dense_7_41072529:("
dense_8_41072550:(
dense_8_41072552:
identity

identity_1¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢dense_8/StatefulPartitionedCallò
conv2d/StatefulPartitionedCallStatefulPartitionedCallxconv2d_41072245conv2d_41072247*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_41072244
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_41072250batch_normalization_41072252batch_normalization_41072254batch_normalization_41072256*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_41071829}
ReluRelu4batch_normalization/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv2d_1_41072271conv2d_1_41072273*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_41072270
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_41072276batch_normalization_1_41072278batch_normalization_1_41072280batch_normalization_1_41072282*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_41071893
Relu_1Relu6batch_normalization_1/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0conv2d_2_41072297conv2d_2_41072299*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_41072296
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_41072302batch_normalization_2_41072304batch_normalization_2_41072306batch_normalization_2_41072308*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_41071957
Relu_2Relu6batch_normalization_2/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0conv2d_3_41072323conv2d_3_41072325*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_41072322
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_41072328batch_normalization_3_41072330batch_normalization_3_41072332batch_normalization_3_41072334*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_41072021
Relu_3Relu6batch_normalization_3/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
flatten/PartitionedCallPartitionedCallRelu_3:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_41072344
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_41072357dense_41072359*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_41072356
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_4_41072362batch_normalization_4_41072364batch_normalization_4_41072366batch_normalization_4_41072368*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_41072087y
Relu_4Relu6batch_normalization_4/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
dropout/PartitionedCallPartitionedCallRelu_4:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072377
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_41072390dense_1_41072392*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_41072389
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_5_41072395batch_normalization_5_41072397batch_normalization_5_41072399batch_normalization_5_41072401*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_41072169y
Relu_5Relu6batch_normalization_5/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
dropout/PartitionedCall_1PartitionedCallRelu_5:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072409
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout/PartitionedCall_1:output:0dense_2_41072423dense_2_41072425*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_41072422Ü
dropout/PartitionedCall_2PartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072432
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout/PartitionedCall_2:output:0dense_3_41072446dense_3_41072448*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_41072445Ü
dropout/PartitionedCall_3PartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072455
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout/PartitionedCall_3:output:0dense_4_41072469dense_4_41072471*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_41072468Ü
dropout/PartitionedCall_4PartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072478
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout/PartitionedCall_4:output:0dense_5_41072492dense_5_41072494*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_41072491
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout/PartitionedCall_1:output:0dense_6_41072509dense_6_41072511*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_41072508Ü
dropout/PartitionedCall_5PartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072432
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout/PartitionedCall_5:output:0dense_7_41072527dense_7_41072529*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_41072526Ü
dropout/PartitionedCall_6PartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072536
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout/PartitionedCall_6:output:0dense_8_41072550dense_8_41072552*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_41072549w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	y

Identity_1Identity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
 

÷
E__inference_dense_2_layer_call_and_return_conditional_losses_41072422

inputs1
matmul_readvariableop_resource:	2-
biasadd_readvariableop_resource:2
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
*
T__inference_tic_tac_toe_neural_net_layer_call_and_return_conditional_losses_41074182
x@
%conv2d_conv2d_readvariableop_resource:5
&conv2d_biasadd_readvariableop_resource:	:
+batch_normalization_readvariableop_resource:	<
-batch_normalization_readvariableop_1_resource:	K
<batch_normalization_fusedbatchnormv3_readvariableop_resource:	M
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_1_conv2d_readvariableop_resource:7
(conv2d_1_biasadd_readvariableop_resource:	<
-batch_normalization_1_readvariableop_resource:	>
/batch_normalization_1_readvariableop_1_resource:	M
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_2_conv2d_readvariableop_resource:7
(conv2d_2_biasadd_readvariableop_resource:	<
-batch_normalization_2_readvariableop_resource:	>
/batch_normalization_2_readvariableop_1_resource:	M
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	C
'conv2d_3_conv2d_readvariableop_resource:7
(conv2d_3_biasadd_readvariableop_resource:	<
-batch_normalization_3_readvariableop_resource:	>
/batch_normalization_3_readvariableop_1_resource:	M
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	O
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	8
$dense_matmul_readvariableop_resource:
4
%dense_biasadd_readvariableop_resource:	F
7batch_normalization_4_batchnorm_readvariableop_resource:	J
;batch_normalization_4_batchnorm_mul_readvariableop_resource:	H
9batch_normalization_4_batchnorm_readvariableop_1_resource:	H
9batch_normalization_4_batchnorm_readvariableop_2_resource:	:
&dense_1_matmul_readvariableop_resource:
6
'dense_1_biasadd_readvariableop_resource:	F
7batch_normalization_5_batchnorm_readvariableop_resource:	J
;batch_normalization_5_batchnorm_mul_readvariableop_resource:	H
9batch_normalization_5_batchnorm_readvariableop_1_resource:	H
9batch_normalization_5_batchnorm_readvariableop_2_resource:	9
&dense_2_matmul_readvariableop_resource:	25
'dense_2_biasadd_readvariableop_resource:28
&dense_3_matmul_readvariableop_resource:25
'dense_3_biasadd_readvariableop_resource:8
&dense_4_matmul_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:8
&dense_5_matmul_readvariableop_resource:	5
'dense_5_biasadd_readvariableop_resource:	9
&dense_6_matmul_readvariableop_resource:	25
'dense_6_biasadd_readvariableop_resource:28
&dense_7_matmul_readvariableop_resource:2(5
'dense_7_biasadd_readvariableop_resource:(8
&dense_8_matmul_readvariableop_resource:(5
'dense_8_biasadd_readvariableop_resource:
identity

identity_1¢3batch_normalization/FusedBatchNormV3/ReadVariableOp¢5batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢"batch_normalization/ReadVariableOp¢$batch_normalization/ReadVariableOp_1¢5batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_1/ReadVariableOp¢&batch_normalization_1/ReadVariableOp_1¢5batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_2/ReadVariableOp¢&batch_normalization_2/ReadVariableOp_1¢5batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_3/ReadVariableOp¢&batch_normalization_3/ReadVariableOp_1¢.batch_normalization_4/batchnorm/ReadVariableOp¢0batch_normalization_4/batchnorm/ReadVariableOp_1¢0batch_normalization_4/batchnorm/ReadVariableOp_2¢2batch_normalization_4/batchnorm/mul/ReadVariableOp¢.batch_normalization_5/batchnorm/ReadVariableOp¢0batch_normalization_5/batchnorm/ReadVariableOp_1¢0batch_normalization_5/batchnorm/ReadVariableOp_2¢2batch_normalization_5/batchnorm/mul/ReadVariableOp¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp¢conv2d_3/BiasAdd/ReadVariableOp¢conv2d_3/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢dense_8/MatMul/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0£
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes	
:*
dtype0
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes	
:*
dtype0­
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0±
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0°
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( q
ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¸
conv2d_1/Conv2DConv2DRelu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0¼
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( u
Relu_1Relu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0º
conv2d_2/Conv2DConv2DRelu_1:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0¼
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( u
Relu_2Relu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0»
conv2d_3/Conv2DConv2DRelu_2:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:*
dtype0
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype0±
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0¼
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( u
Relu_3Relu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   {
flatten/ReshapeReshapeRelu_3:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0j
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:}
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:«
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0·
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
: 
%batch_normalization_4/batchnorm/mul_1Muldense/BiasAdd:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0µ
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:§
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0µ
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:µ
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
Relu_4Relu)batch_normalization_4/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
dropout/IdentityIdentityRelu_4:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0j
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
#batch_normalization_5/batchnorm/addAddV26batch_normalization_5/batchnorm/ReadVariableOp:value:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes	
:}
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes	
:«
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0·
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:¢
%batch_normalization_5/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0µ
%batch_normalization_5/batchnorm/mul_2Mul8batch_normalization_5/batchnorm/ReadVariableOp_1:value:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes	
:§
0batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0µ
#batch_normalization_5/batchnorm/subSub8batch_normalization_5/batchnorm/ReadVariableOp_2:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:µ
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
Relu_5Relu)batch_normalization_5/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
dropout/Identity_1IdentityRelu_5:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype0
dense_2/MatMulMatMuldropout/Identity_1:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2l
dropout/Identity_2Identitydense_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
dense_3/MatMulMatMuldropout/Identity_2:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
dropout/Identity_3Identitydense_3/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_4/MatMulMatMuldropout/Identity_3:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
dropout/Identity_4Identitydense_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:	*
dtype0
dense_5/MatMulMatMuldropout/Identity_4:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	f
dense_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype0
dense_6/MatMulMatMuldropout/Identity_1:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2l
dropout/Identity_5Identitydense_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype0
dense_7/MatMulMatMuldropout/Identity_5:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(l
dropout/Identity_6Identitydense_7/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0
dense_8/MatMulMatMuldropout/Identity_6:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_8/TanhTanhdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_5/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	a

Identity_1Identitydense_8/Tanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1/^batch_normalization_4/batchnorm/ReadVariableOp1^batch_normalization_4/batchnorm/ReadVariableOp_11^batch_normalization_4/batchnorm/ReadVariableOp_23^batch_normalization_4/batchnorm/mul/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp1^batch_normalization_5/batchnorm/ReadVariableOp_11^batch_normalization_5/batchnorm/ReadVariableOp_23^batch_normalization_5/batchnorm/mul/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2d
0batch_normalization_4/batchnorm/ReadVariableOp_10batch_normalization_4/batchnorm/ReadVariableOp_12d
0batch_normalization_4/batchnorm/ReadVariableOp_20batch_normalization_4/batchnorm/ReadVariableOp_22h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp2`
.batch_normalization_5/batchnorm/ReadVariableOp.batch_normalization_5/batchnorm/ReadVariableOp2d
0batch_normalization_5/batchnorm/ReadVariableOp_10batch_normalization_5/batchnorm/ReadVariableOp_12d
0batch_normalization_5/batchnorm/ReadVariableOp_20batch_normalization_5/batchnorm/ReadVariableOp_22h
2batch_normalization_5/batchnorm/mul/ReadVariableOp2batch_normalization_5/batchnorm/mul/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
´


F__inference_conv2d_2_layer_call_and_return_conditional_losses_41072296

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é
Ý
T__inference_tic_tac_toe_neural_net_layer_call_and_return_conditional_losses_41073666
input_1*
conv2d_41073531:
conv2d_41073533:	+
batch_normalization_41073536:	+
batch_normalization_41073538:	+
batch_normalization_41073540:	+
batch_normalization_41073542:	-
conv2d_1_41073546: 
conv2d_1_41073548:	-
batch_normalization_1_41073551:	-
batch_normalization_1_41073553:	-
batch_normalization_1_41073555:	-
batch_normalization_1_41073557:	-
conv2d_2_41073561: 
conv2d_2_41073563:	-
batch_normalization_2_41073566:	-
batch_normalization_2_41073568:	-
batch_normalization_2_41073570:	-
batch_normalization_2_41073572:	-
conv2d_3_41073576: 
conv2d_3_41073578:	-
batch_normalization_3_41073581:	-
batch_normalization_3_41073583:	-
batch_normalization_3_41073585:	-
batch_normalization_3_41073587:	"
dense_41073592:

dense_41073594:	-
batch_normalization_4_41073597:	-
batch_normalization_4_41073599:	-
batch_normalization_4_41073601:	-
batch_normalization_4_41073603:	$
dense_1_41073608:

dense_1_41073610:	-
batch_normalization_5_41073613:	-
batch_normalization_5_41073615:	-
batch_normalization_5_41073617:	-
batch_normalization_5_41073619:	#
dense_2_41073624:	2
dense_2_41073626:2"
dense_3_41073630:2
dense_3_41073632:"
dense_4_41073636:
dense_4_41073638:"
dense_5_41073642:	
dense_5_41073644:	#
dense_6_41073647:	2
dense_6_41073649:2"
dense_7_41073653:2(
dense_7_41073655:("
dense_8_41073659:(
dense_8_41073661:
identity

identity_1¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout/StatefulPartitionedCall_1¢!dropout/StatefulPartitionedCall_2¢!dropout/StatefulPartitionedCall_3¢!dropout/StatefulPartitionedCall_4¢!dropout/StatefulPartitionedCall_5¢!dropout/StatefulPartitionedCall_6ø
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_41073531conv2d_41073533*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_41072244
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_41073536batch_normalization_41073538batch_normalization_41073540batch_normalization_41073542*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_41071860}
ReluRelu4batch_normalization/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv2d_1_41073546conv2d_1_41073548*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_41072270
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_41073551batch_normalization_1_41073553batch_normalization_1_41073555batch_normalization_1_41073557*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_41071924
Relu_1Relu6batch_normalization_1/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0conv2d_2_41073561conv2d_2_41073563*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_41072296
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_41073566batch_normalization_2_41073568batch_normalization_2_41073570batch_normalization_2_41073572*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_41071988
Relu_2Relu6batch_normalization_2/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0conv2d_3_41073576conv2d_3_41073578*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_41072322
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_41073581batch_normalization_3_41073583batch_normalization_3_41073585batch_normalization_3_41073587*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_41072052
Relu_3Relu6batch_normalization_3/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
flatten/PartitionedCallPartitionedCallRelu_3:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_41072344
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_41073592dense_41073594*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_41072356
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_4_41073597batch_normalization_4_41073599batch_normalization_4_41073601batch_normalization_4_41073603*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_41072134y
Relu_4Relu6batch_normalization_4/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
dropout/StatefulPartitionedCallStatefulPartitionedCallRelu_4:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072872
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_41073608dense_1_41073610*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_41072389
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_5_41073613batch_normalization_5_41073615batch_normalization_5_41073617batch_normalization_5_41073619*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_41072216y
Relu_5Relu6batch_normalization_5/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿû
!dropout/StatefulPartitionedCall_1StatefulPartitionedCallRelu_5:activations:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072840
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout/StatefulPartitionedCall_1:output:0dense_2_41073624dense_2_41073626*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_41072422
!dropout/StatefulPartitionedCall_2StatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0"^dropout/StatefulPartitionedCall_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072724
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout/StatefulPartitionedCall_2:output:0dense_3_41073630dense_3_41073632*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_41072445
!dropout/StatefulPartitionedCall_3StatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0"^dropout/StatefulPartitionedCall_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072798
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout/StatefulPartitionedCall_3:output:0dense_4_41073636dense_4_41073638*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_41072468
!dropout/StatefulPartitionedCall_4StatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0"^dropout/StatefulPartitionedCall_3*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072766
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout/StatefulPartitionedCall_4:output:0dense_5_41073642dense_5_41073644*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_41072491
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout/StatefulPartitionedCall_1:output:0dense_6_41073647dense_6_41073649*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_41072508
!dropout/StatefulPartitionedCall_5StatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0"^dropout/StatefulPartitionedCall_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072724
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout/StatefulPartitionedCall_5:output:0dense_7_41073653dense_7_41073655*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_41072526
!dropout/StatefulPartitionedCall_6StatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0"^dropout/StatefulPartitionedCall_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072692
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout/StatefulPartitionedCall_6:output:0dense_8_41073659dense_8_41073661*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_41072549w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	y

Identity_1Identity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout/StatefulPartitionedCall_1"^dropout/StatefulPartitionedCall_2"^dropout/StatefulPartitionedCall_3"^dropout/StatefulPartitionedCall_4"^dropout/StatefulPartitionedCall_5"^dropout/StatefulPartitionedCall_6*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout/StatefulPartitionedCall_1!dropout/StatefulPartitionedCall_12F
!dropout/StatefulPartitionedCall_2!dropout/StatefulPartitionedCall_22F
!dropout/StatefulPartitionedCall_3!dropout/StatefulPartitionedCall_32F
!dropout/StatefulPartitionedCall_4!dropout/StatefulPartitionedCall_42F
!dropout/StatefulPartitionedCall_5!dropout/StatefulPartitionedCall_52F
!dropout/StatefulPartitionedCall_6!dropout/StatefulPartitionedCall_6:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
 

÷
E__inference_dense_2_layer_call_and_return_conditional_losses_41075167

inputs1
matmul_readvariableop_resource:	2-
biasadd_readvariableop_resource:2
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ö
E__inference_dense_3_layer_call_and_return_conditional_losses_41072445

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
 

÷
E__inference_dense_6_layer_call_and_return_conditional_losses_41072508

inputs1
matmul_readvariableop_resource:	2-
biasadd_readvariableop_resource:2
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

*__inference_dense_7_layer_call_fn_41075256

inputs
unknown:2(
	unknown_0:(
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_41072526o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
®

ÿ
D__inference_conv2d_layer_call_and_return_conditional_losses_41072244

inputs9
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
c
E__inference_dropout_layer_call_and_return_conditional_losses_41072432

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
	
×
8__inference_batch_normalization_2_layer_call_fn_41074646

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_41071957
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó	
ù
E__inference_dense_1_layer_call_and_return_conditional_losses_41072389

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±%
ð
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_41072134

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ö
E__inference_dense_3_layer_call_and_return_conditional_losses_41075187

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
É
a
E__inference_flatten_layer_call_and_return_conditional_losses_41072344

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
×
8__inference_batch_normalization_5_layer_call_fn_41074918

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_41072169p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó	
d
E__inference_dropout_layer_call_and_return_conditional_losses_41075087

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
ó	
d
E__inference_dropout_layer_call_and_return_conditional_losses_41072724

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs


ö
E__inference_dense_4_layer_call_and_return_conditional_losses_41075207

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
c
E__inference_dropout_layer_call_and_return_conditional_losses_41075060

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
õ
c
*__inference_dropout_layer_call_fn_41075045

inputs
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072872p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó	
d
E__inference_dropout_layer_call_and_return_conditional_losses_41075123

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó	
d
E__inference_dropout_layer_call_and_return_conditional_losses_41072766

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Æ
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_41072052

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î
 
)__inference_conv2d_layer_call_fn_41074461

inputs"
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_41072244x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Æ
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_41074776

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
c
E__inference_dropout_layer_call_and_return_conditional_losses_41075075

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ([

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ("!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs

F
*__inference_dropout_layer_call_fn_41075020

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072455`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î

9__inference_tic_tac_toe_neural_net_layer_call_fn_41073390
input_1"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:


unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:


unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	

unknown_34:	

unknown_35:	2

unknown_36:2

unknown_37:2

unknown_38:

unknown_39:

unknown_40:

unknown_41:	

unknown_42:	

unknown_43:	2

unknown_44:2

unknown_45:2(

unknown_46:(

unknown_47:(

unknown_48:
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ	:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 #$%&'()*+,-./012*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_tic_tac_toe_neural_net_layer_call_and_return_conditional_losses_41073178o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
²
×
8__inference_batch_normalization_4_layer_call_fn_41074832

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_41072134p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û	
d
E__inference_dropout_layer_call_and_return_conditional_losses_41072872

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ûÍ
ä;
#__inference__wrapped_model_41071806
input_1W
<tic_tac_toe_neural_net_conv2d_conv2d_readvariableop_resource:L
=tic_tac_toe_neural_net_conv2d_biasadd_readvariableop_resource:	Q
Btic_tac_toe_neural_net_batch_normalization_readvariableop_resource:	S
Dtic_tac_toe_neural_net_batch_normalization_readvariableop_1_resource:	b
Stic_tac_toe_neural_net_batch_normalization_fusedbatchnormv3_readvariableop_resource:	d
Utic_tac_toe_neural_net_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	Z
>tic_tac_toe_neural_net_conv2d_1_conv2d_readvariableop_resource:N
?tic_tac_toe_neural_net_conv2d_1_biasadd_readvariableop_resource:	S
Dtic_tac_toe_neural_net_batch_normalization_1_readvariableop_resource:	U
Ftic_tac_toe_neural_net_batch_normalization_1_readvariableop_1_resource:	d
Utic_tac_toe_neural_net_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	f
Wtic_tac_toe_neural_net_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	Z
>tic_tac_toe_neural_net_conv2d_2_conv2d_readvariableop_resource:N
?tic_tac_toe_neural_net_conv2d_2_biasadd_readvariableop_resource:	S
Dtic_tac_toe_neural_net_batch_normalization_2_readvariableop_resource:	U
Ftic_tac_toe_neural_net_batch_normalization_2_readvariableop_1_resource:	d
Utic_tac_toe_neural_net_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	f
Wtic_tac_toe_neural_net_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	Z
>tic_tac_toe_neural_net_conv2d_3_conv2d_readvariableop_resource:N
?tic_tac_toe_neural_net_conv2d_3_biasadd_readvariableop_resource:	S
Dtic_tac_toe_neural_net_batch_normalization_3_readvariableop_resource:	U
Ftic_tac_toe_neural_net_batch_normalization_3_readvariableop_1_resource:	d
Utic_tac_toe_neural_net_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	f
Wtic_tac_toe_neural_net_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	O
;tic_tac_toe_neural_net_dense_matmul_readvariableop_resource:
K
<tic_tac_toe_neural_net_dense_biasadd_readvariableop_resource:	]
Ntic_tac_toe_neural_net_batch_normalization_4_batchnorm_readvariableop_resource:	a
Rtic_tac_toe_neural_net_batch_normalization_4_batchnorm_mul_readvariableop_resource:	_
Ptic_tac_toe_neural_net_batch_normalization_4_batchnorm_readvariableop_1_resource:	_
Ptic_tac_toe_neural_net_batch_normalization_4_batchnorm_readvariableop_2_resource:	Q
=tic_tac_toe_neural_net_dense_1_matmul_readvariableop_resource:
M
>tic_tac_toe_neural_net_dense_1_biasadd_readvariableop_resource:	]
Ntic_tac_toe_neural_net_batch_normalization_5_batchnorm_readvariableop_resource:	a
Rtic_tac_toe_neural_net_batch_normalization_5_batchnorm_mul_readvariableop_resource:	_
Ptic_tac_toe_neural_net_batch_normalization_5_batchnorm_readvariableop_1_resource:	_
Ptic_tac_toe_neural_net_batch_normalization_5_batchnorm_readvariableop_2_resource:	P
=tic_tac_toe_neural_net_dense_2_matmul_readvariableop_resource:	2L
>tic_tac_toe_neural_net_dense_2_biasadd_readvariableop_resource:2O
=tic_tac_toe_neural_net_dense_3_matmul_readvariableop_resource:2L
>tic_tac_toe_neural_net_dense_3_biasadd_readvariableop_resource:O
=tic_tac_toe_neural_net_dense_4_matmul_readvariableop_resource:L
>tic_tac_toe_neural_net_dense_4_biasadd_readvariableop_resource:O
=tic_tac_toe_neural_net_dense_5_matmul_readvariableop_resource:	L
>tic_tac_toe_neural_net_dense_5_biasadd_readvariableop_resource:	P
=tic_tac_toe_neural_net_dense_6_matmul_readvariableop_resource:	2L
>tic_tac_toe_neural_net_dense_6_biasadd_readvariableop_resource:2O
=tic_tac_toe_neural_net_dense_7_matmul_readvariableop_resource:2(L
>tic_tac_toe_neural_net_dense_7_biasadd_readvariableop_resource:(O
=tic_tac_toe_neural_net_dense_8_matmul_readvariableop_resource:(L
>tic_tac_toe_neural_net_dense_8_biasadd_readvariableop_resource:
identity

identity_1¢Jtic_tac_toe_neural_net/batch_normalization/FusedBatchNormV3/ReadVariableOp¢Ltic_tac_toe_neural_net/batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢9tic_tac_toe_neural_net/batch_normalization/ReadVariableOp¢;tic_tac_toe_neural_net/batch_normalization/ReadVariableOp_1¢Ltic_tac_toe_neural_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢Ntic_tac_toe_neural_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢;tic_tac_toe_neural_net/batch_normalization_1/ReadVariableOp¢=tic_tac_toe_neural_net/batch_normalization_1/ReadVariableOp_1¢Ltic_tac_toe_neural_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOp¢Ntic_tac_toe_neural_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¢;tic_tac_toe_neural_net/batch_normalization_2/ReadVariableOp¢=tic_tac_toe_neural_net/batch_normalization_2/ReadVariableOp_1¢Ltic_tac_toe_neural_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢Ntic_tac_toe_neural_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢;tic_tac_toe_neural_net/batch_normalization_3/ReadVariableOp¢=tic_tac_toe_neural_net/batch_normalization_3/ReadVariableOp_1¢Etic_tac_toe_neural_net/batch_normalization_4/batchnorm/ReadVariableOp¢Gtic_tac_toe_neural_net/batch_normalization_4/batchnorm/ReadVariableOp_1¢Gtic_tac_toe_neural_net/batch_normalization_4/batchnorm/ReadVariableOp_2¢Itic_tac_toe_neural_net/batch_normalization_4/batchnorm/mul/ReadVariableOp¢Etic_tac_toe_neural_net/batch_normalization_5/batchnorm/ReadVariableOp¢Gtic_tac_toe_neural_net/batch_normalization_5/batchnorm/ReadVariableOp_1¢Gtic_tac_toe_neural_net/batch_normalization_5/batchnorm/ReadVariableOp_2¢Itic_tac_toe_neural_net/batch_normalization_5/batchnorm/mul/ReadVariableOp¢4tic_tac_toe_neural_net/conv2d/BiasAdd/ReadVariableOp¢3tic_tac_toe_neural_net/conv2d/Conv2D/ReadVariableOp¢6tic_tac_toe_neural_net/conv2d_1/BiasAdd/ReadVariableOp¢5tic_tac_toe_neural_net/conv2d_1/Conv2D/ReadVariableOp¢6tic_tac_toe_neural_net/conv2d_2/BiasAdd/ReadVariableOp¢5tic_tac_toe_neural_net/conv2d_2/Conv2D/ReadVariableOp¢6tic_tac_toe_neural_net/conv2d_3/BiasAdd/ReadVariableOp¢5tic_tac_toe_neural_net/conv2d_3/Conv2D/ReadVariableOp¢3tic_tac_toe_neural_net/dense/BiasAdd/ReadVariableOp¢2tic_tac_toe_neural_net/dense/MatMul/ReadVariableOp¢5tic_tac_toe_neural_net/dense_1/BiasAdd/ReadVariableOp¢4tic_tac_toe_neural_net/dense_1/MatMul/ReadVariableOp¢5tic_tac_toe_neural_net/dense_2/BiasAdd/ReadVariableOp¢4tic_tac_toe_neural_net/dense_2/MatMul/ReadVariableOp¢5tic_tac_toe_neural_net/dense_3/BiasAdd/ReadVariableOp¢4tic_tac_toe_neural_net/dense_3/MatMul/ReadVariableOp¢5tic_tac_toe_neural_net/dense_4/BiasAdd/ReadVariableOp¢4tic_tac_toe_neural_net/dense_4/MatMul/ReadVariableOp¢5tic_tac_toe_neural_net/dense_5/BiasAdd/ReadVariableOp¢4tic_tac_toe_neural_net/dense_5/MatMul/ReadVariableOp¢5tic_tac_toe_neural_net/dense_6/BiasAdd/ReadVariableOp¢4tic_tac_toe_neural_net/dense_6/MatMul/ReadVariableOp¢5tic_tac_toe_neural_net/dense_7/BiasAdd/ReadVariableOp¢4tic_tac_toe_neural_net/dense_7/MatMul/ReadVariableOp¢5tic_tac_toe_neural_net/dense_8/BiasAdd/ReadVariableOp¢4tic_tac_toe_neural_net/dense_8/MatMul/ReadVariableOp¹
3tic_tac_toe_neural_net/conv2d/Conv2D/ReadVariableOpReadVariableOp<tic_tac_toe_neural_net_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0×
$tic_tac_toe_neural_net/conv2d/Conv2DConv2Dinput_1;tic_tac_toe_neural_net/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¯
4tic_tac_toe_neural_net/conv2d/BiasAdd/ReadVariableOpReadVariableOp=tic_tac_toe_neural_net_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ø
%tic_tac_toe_neural_net/conv2d/BiasAddBiasAdd-tic_tac_toe_neural_net/conv2d/Conv2D:output:0<tic_tac_toe_neural_net/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
9tic_tac_toe_neural_net/batch_normalization/ReadVariableOpReadVariableOpBtic_tac_toe_neural_net_batch_normalization_readvariableop_resource*
_output_shapes	
:*
dtype0½
;tic_tac_toe_neural_net/batch_normalization/ReadVariableOp_1ReadVariableOpDtic_tac_toe_neural_net_batch_normalization_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
Jtic_tac_toe_neural_net/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpStic_tac_toe_neural_net_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0ß
Ltic_tac_toe_neural_net/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpUtic_tac_toe_neural_net_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0º
;tic_tac_toe_neural_net/batch_normalization/FusedBatchNormV3FusedBatchNormV3.tic_tac_toe_neural_net/conv2d/BiasAdd:output:0Atic_tac_toe_neural_net/batch_normalization/ReadVariableOp:value:0Ctic_tac_toe_neural_net/batch_normalization/ReadVariableOp_1:value:0Rtic_tac_toe_neural_net/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Ttic_tac_toe_neural_net/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 
tic_tac_toe_neural_net/ReluRelu?tic_tac_toe_neural_net/batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
5tic_tac_toe_neural_net/conv2d_1/Conv2D/ReadVariableOpReadVariableOp>tic_tac_toe_neural_net_conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
&tic_tac_toe_neural_net/conv2d_1/Conv2DConv2D)tic_tac_toe_neural_net/Relu:activations:0=tic_tac_toe_neural_net/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
³
6tic_tac_toe_neural_net/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp?tic_tac_toe_neural_net_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Þ
'tic_tac_toe_neural_net/conv2d_1/BiasAddBiasAdd/tic_tac_toe_neural_net/conv2d_1/Conv2D:output:0>tic_tac_toe_neural_net/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
;tic_tac_toe_neural_net/batch_normalization_1/ReadVariableOpReadVariableOpDtic_tac_toe_neural_net_batch_normalization_1_readvariableop_resource*
_output_shapes	
:*
dtype0Á
=tic_tac_toe_neural_net/batch_normalization_1/ReadVariableOp_1ReadVariableOpFtic_tac_toe_neural_net_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:*
dtype0ß
Ltic_tac_toe_neural_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpUtic_tac_toe_neural_net_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0ã
Ntic_tac_toe_neural_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWtic_tac_toe_neural_net_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Æ
=tic_tac_toe_neural_net/batch_normalization_1/FusedBatchNormV3FusedBatchNormV30tic_tac_toe_neural_net/conv2d_1/BiasAdd:output:0Ctic_tac_toe_neural_net/batch_normalization_1/ReadVariableOp:value:0Etic_tac_toe_neural_net/batch_normalization_1/ReadVariableOp_1:value:0Ttic_tac_toe_neural_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Vtic_tac_toe_neural_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( £
tic_tac_toe_neural_net/Relu_1ReluAtic_tac_toe_neural_net/batch_normalization_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
5tic_tac_toe_neural_net/conv2d_2/Conv2D/ReadVariableOpReadVariableOp>tic_tac_toe_neural_net_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ÿ
&tic_tac_toe_neural_net/conv2d_2/Conv2DConv2D+tic_tac_toe_neural_net/Relu_1:activations:0=tic_tac_toe_neural_net/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
³
6tic_tac_toe_neural_net/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp?tic_tac_toe_neural_net_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Þ
'tic_tac_toe_neural_net/conv2d_2/BiasAddBiasAdd/tic_tac_toe_neural_net/conv2d_2/Conv2D:output:0>tic_tac_toe_neural_net/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
;tic_tac_toe_neural_net/batch_normalization_2/ReadVariableOpReadVariableOpDtic_tac_toe_neural_net_batch_normalization_2_readvariableop_resource*
_output_shapes	
:*
dtype0Á
=tic_tac_toe_neural_net/batch_normalization_2/ReadVariableOp_1ReadVariableOpFtic_tac_toe_neural_net_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:*
dtype0ß
Ltic_tac_toe_neural_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpUtic_tac_toe_neural_net_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0ã
Ntic_tac_toe_neural_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWtic_tac_toe_neural_net_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Æ
=tic_tac_toe_neural_net/batch_normalization_2/FusedBatchNormV3FusedBatchNormV30tic_tac_toe_neural_net/conv2d_2/BiasAdd:output:0Ctic_tac_toe_neural_net/batch_normalization_2/ReadVariableOp:value:0Etic_tac_toe_neural_net/batch_normalization_2/ReadVariableOp_1:value:0Ttic_tac_toe_neural_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Vtic_tac_toe_neural_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( £
tic_tac_toe_neural_net/Relu_2ReluAtic_tac_toe_neural_net/batch_normalization_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
5tic_tac_toe_neural_net/conv2d_3/Conv2D/ReadVariableOpReadVariableOp>tic_tac_toe_neural_net_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
&tic_tac_toe_neural_net/conv2d_3/Conv2DConv2D+tic_tac_toe_neural_net/Relu_2:activations:0=tic_tac_toe_neural_net/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
³
6tic_tac_toe_neural_net/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp?tic_tac_toe_neural_net_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Þ
'tic_tac_toe_neural_net/conv2d_3/BiasAddBiasAdd/tic_tac_toe_neural_net/conv2d_3/Conv2D:output:0>tic_tac_toe_neural_net/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
;tic_tac_toe_neural_net/batch_normalization_3/ReadVariableOpReadVariableOpDtic_tac_toe_neural_net_batch_normalization_3_readvariableop_resource*
_output_shapes	
:*
dtype0Á
=tic_tac_toe_neural_net/batch_normalization_3/ReadVariableOp_1ReadVariableOpFtic_tac_toe_neural_net_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ß
Ltic_tac_toe_neural_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpUtic_tac_toe_neural_net_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0ã
Ntic_tac_toe_neural_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpWtic_tac_toe_neural_net_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Æ
=tic_tac_toe_neural_net/batch_normalization_3/FusedBatchNormV3FusedBatchNormV30tic_tac_toe_neural_net/conv2d_3/BiasAdd:output:0Ctic_tac_toe_neural_net/batch_normalization_3/ReadVariableOp:value:0Etic_tac_toe_neural_net/batch_normalization_3/ReadVariableOp_1:value:0Ttic_tac_toe_neural_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Vtic_tac_toe_neural_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( £
tic_tac_toe_neural_net/Relu_3ReluAtic_tac_toe_neural_net/batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
$tic_tac_toe_neural_net/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   À
&tic_tac_toe_neural_net/flatten/ReshapeReshape+tic_tac_toe_neural_net/Relu_3:activations:0-tic_tac_toe_neural_net/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
2tic_tac_toe_neural_net/dense/MatMul/ReadVariableOpReadVariableOp;tic_tac_toe_neural_net_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Í
#tic_tac_toe_neural_net/dense/MatMulMatMul/tic_tac_toe_neural_net/flatten/Reshape:output:0:tic_tac_toe_neural_net/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
3tic_tac_toe_neural_net/dense/BiasAdd/ReadVariableOpReadVariableOp<tic_tac_toe_neural_net_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Î
$tic_tac_toe_neural_net/dense/BiasAddBiasAdd-tic_tac_toe_neural_net/dense/MatMul:product:0;tic_tac_toe_neural_net/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
Etic_tac_toe_neural_net/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOpNtic_tac_toe_neural_net_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0
<tic_tac_toe_neural_net/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:ÿ
:tic_tac_toe_neural_net/batch_normalization_4/batchnorm/addAddV2Mtic_tac_toe_neural_net/batch_normalization_4/batchnorm/ReadVariableOp:value:0Etic_tac_toe_neural_net/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes	
:«
<tic_tac_toe_neural_net/batch_normalization_4/batchnorm/RsqrtRsqrt>tic_tac_toe_neural_net/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes	
:Ù
Itic_tac_toe_neural_net/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpRtic_tac_toe_neural_net_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0ü
:tic_tac_toe_neural_net/batch_normalization_4/batchnorm/mulMul@tic_tac_toe_neural_net/batch_normalization_4/batchnorm/Rsqrt:y:0Qtic_tac_toe_neural_net/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:å
<tic_tac_toe_neural_net/batch_normalization_4/batchnorm/mul_1Mul-tic_tac_toe_neural_net/dense/BiasAdd:output:0>tic_tac_toe_neural_net/batch_normalization_4/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
Gtic_tac_toe_neural_net/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpPtic_tac_toe_neural_net_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0ú
<tic_tac_toe_neural_net/batch_normalization_4/batchnorm/mul_2MulOtic_tac_toe_neural_net/batch_normalization_4/batchnorm/ReadVariableOp_1:value:0>tic_tac_toe_neural_net/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes	
:Õ
Gtic_tac_toe_neural_net/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpPtic_tac_toe_neural_net_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0ú
:tic_tac_toe_neural_net/batch_normalization_4/batchnorm/subSubOtic_tac_toe_neural_net/batch_normalization_4/batchnorm/ReadVariableOp_2:value:0@tic_tac_toe_neural_net/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ú
<tic_tac_toe_neural_net/batch_normalization_4/batchnorm/add_1AddV2@tic_tac_toe_neural_net/batch_normalization_4/batchnorm/mul_1:z:0>tic_tac_toe_neural_net/batch_normalization_4/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
tic_tac_toe_neural_net/Relu_4Relu@tic_tac_toe_neural_net/batch_normalization_4/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'tic_tac_toe_neural_net/dropout/IdentityIdentity+tic_tac_toe_neural_net/Relu_4:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
4tic_tac_toe_neural_net/dense_1/MatMul/ReadVariableOpReadVariableOp=tic_tac_toe_neural_net_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ò
%tic_tac_toe_neural_net/dense_1/MatMulMatMul0tic_tac_toe_neural_net/dropout/Identity:output:0<tic_tac_toe_neural_net/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
5tic_tac_toe_neural_net/dense_1/BiasAdd/ReadVariableOpReadVariableOp>tic_tac_toe_neural_net_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ô
&tic_tac_toe_neural_net/dense_1/BiasAddBiasAdd/tic_tac_toe_neural_net/dense_1/MatMul:product:0=tic_tac_toe_neural_net/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
Etic_tac_toe_neural_net/batch_normalization_5/batchnorm/ReadVariableOpReadVariableOpNtic_tac_toe_neural_net_batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0
<tic_tac_toe_neural_net/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:ÿ
:tic_tac_toe_neural_net/batch_normalization_5/batchnorm/addAddV2Mtic_tac_toe_neural_net/batch_normalization_5/batchnorm/ReadVariableOp:value:0Etic_tac_toe_neural_net/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes	
:«
<tic_tac_toe_neural_net/batch_normalization_5/batchnorm/RsqrtRsqrt>tic_tac_toe_neural_net/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes	
:Ù
Itic_tac_toe_neural_net/batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpRtic_tac_toe_neural_net_batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0ü
:tic_tac_toe_neural_net/batch_normalization_5/batchnorm/mulMul@tic_tac_toe_neural_net/batch_normalization_5/batchnorm/Rsqrt:y:0Qtic_tac_toe_neural_net/batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ç
<tic_tac_toe_neural_net/batch_normalization_5/batchnorm/mul_1Mul/tic_tac_toe_neural_net/dense_1/BiasAdd:output:0>tic_tac_toe_neural_net/batch_normalization_5/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
Gtic_tac_toe_neural_net/batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOpPtic_tac_toe_neural_net_batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0ú
<tic_tac_toe_neural_net/batch_normalization_5/batchnorm/mul_2MulOtic_tac_toe_neural_net/batch_normalization_5/batchnorm/ReadVariableOp_1:value:0>tic_tac_toe_neural_net/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes	
:Õ
Gtic_tac_toe_neural_net/batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOpPtic_tac_toe_neural_net_batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0ú
:tic_tac_toe_neural_net/batch_normalization_5/batchnorm/subSubOtic_tac_toe_neural_net/batch_normalization_5/batchnorm/ReadVariableOp_2:value:0@tic_tac_toe_neural_net/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ú
<tic_tac_toe_neural_net/batch_normalization_5/batchnorm/add_1AddV2@tic_tac_toe_neural_net/batch_normalization_5/batchnorm/mul_1:z:0>tic_tac_toe_neural_net/batch_normalization_5/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
tic_tac_toe_neural_net/Relu_5Relu@tic_tac_toe_neural_net/batch_normalization_5/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)tic_tac_toe_neural_net/dropout/Identity_1Identity+tic_tac_toe_neural_net/Relu_5:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
4tic_tac_toe_neural_net/dense_2/MatMul/ReadVariableOpReadVariableOp=tic_tac_toe_neural_net_dense_2_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype0Ó
%tic_tac_toe_neural_net/dense_2/MatMulMatMul2tic_tac_toe_neural_net/dropout/Identity_1:output:0<tic_tac_toe_neural_net/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2°
5tic_tac_toe_neural_net/dense_2/BiasAdd/ReadVariableOpReadVariableOp>tic_tac_toe_neural_net_dense_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0Ó
&tic_tac_toe_neural_net/dense_2/BiasAddBiasAdd/tic_tac_toe_neural_net/dense_2/MatMul:product:0=tic_tac_toe_neural_net/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
#tic_tac_toe_neural_net/dense_2/ReluRelu/tic_tac_toe_neural_net/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
)tic_tac_toe_neural_net/dropout/Identity_2Identity1tic_tac_toe_neural_net/dense_2/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2²
4tic_tac_toe_neural_net/dense_3/MatMul/ReadVariableOpReadVariableOp=tic_tac_toe_neural_net_dense_3_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0Ó
%tic_tac_toe_neural_net/dense_3/MatMulMatMul2tic_tac_toe_neural_net/dropout/Identity_2:output:0<tic_tac_toe_neural_net/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
5tic_tac_toe_neural_net/dense_3/BiasAdd/ReadVariableOpReadVariableOp>tic_tac_toe_neural_net_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ó
&tic_tac_toe_neural_net/dense_3/BiasAddBiasAdd/tic_tac_toe_neural_net/dense_3/MatMul:product:0=tic_tac_toe_neural_net/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#tic_tac_toe_neural_net/dense_3/ReluRelu/tic_tac_toe_neural_net/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)tic_tac_toe_neural_net/dropout/Identity_3Identity1tic_tac_toe_neural_net/dense_3/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
4tic_tac_toe_neural_net/dense_4/MatMul/ReadVariableOpReadVariableOp=tic_tac_toe_neural_net_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ó
%tic_tac_toe_neural_net/dense_4/MatMulMatMul2tic_tac_toe_neural_net/dropout/Identity_3:output:0<tic_tac_toe_neural_net/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
5tic_tac_toe_neural_net/dense_4/BiasAdd/ReadVariableOpReadVariableOp>tic_tac_toe_neural_net_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ó
&tic_tac_toe_neural_net/dense_4/BiasAddBiasAdd/tic_tac_toe_neural_net/dense_4/MatMul:product:0=tic_tac_toe_neural_net/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#tic_tac_toe_neural_net/dense_4/ReluRelu/tic_tac_toe_neural_net/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)tic_tac_toe_neural_net/dropout/Identity_4Identity1tic_tac_toe_neural_net/dense_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
4tic_tac_toe_neural_net/dense_5/MatMul/ReadVariableOpReadVariableOp=tic_tac_toe_neural_net_dense_5_matmul_readvariableop_resource*
_output_shapes

:	*
dtype0Ó
%tic_tac_toe_neural_net/dense_5/MatMulMatMul2tic_tac_toe_neural_net/dropout/Identity_4:output:0<tic_tac_toe_neural_net/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	°
5tic_tac_toe_neural_net/dense_5/BiasAdd/ReadVariableOpReadVariableOp>tic_tac_toe_neural_net_dense_5_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0Ó
&tic_tac_toe_neural_net/dense_5/BiasAddBiasAdd/tic_tac_toe_neural_net/dense_5/MatMul:product:0=tic_tac_toe_neural_net/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
&tic_tac_toe_neural_net/dense_5/SoftmaxSoftmax/tic_tac_toe_neural_net/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	³
4tic_tac_toe_neural_net/dense_6/MatMul/ReadVariableOpReadVariableOp=tic_tac_toe_neural_net_dense_6_matmul_readvariableop_resource*
_output_shapes
:	2*
dtype0Ó
%tic_tac_toe_neural_net/dense_6/MatMulMatMul2tic_tac_toe_neural_net/dropout/Identity_1:output:0<tic_tac_toe_neural_net/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2°
5tic_tac_toe_neural_net/dense_6/BiasAdd/ReadVariableOpReadVariableOp>tic_tac_toe_neural_net_dense_6_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0Ó
&tic_tac_toe_neural_net/dense_6/BiasAddBiasAdd/tic_tac_toe_neural_net/dense_6/MatMul:product:0=tic_tac_toe_neural_net/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
#tic_tac_toe_neural_net/dense_6/ReluRelu/tic_tac_toe_neural_net/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
)tic_tac_toe_neural_net/dropout/Identity_5Identity1tic_tac_toe_neural_net/dense_6/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2²
4tic_tac_toe_neural_net/dense_7/MatMul/ReadVariableOpReadVariableOp=tic_tac_toe_neural_net_dense_7_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype0Ó
%tic_tac_toe_neural_net/dense_7/MatMulMatMul2tic_tac_toe_neural_net/dropout/Identity_5:output:0<tic_tac_toe_neural_net/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(°
5tic_tac_toe_neural_net/dense_7/BiasAdd/ReadVariableOpReadVariableOp>tic_tac_toe_neural_net_dense_7_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0Ó
&tic_tac_toe_neural_net/dense_7/BiasAddBiasAdd/tic_tac_toe_neural_net/dense_7/MatMul:product:0=tic_tac_toe_neural_net/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
#tic_tac_toe_neural_net/dense_7/ReluRelu/tic_tac_toe_neural_net/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
)tic_tac_toe_neural_net/dropout/Identity_6Identity1tic_tac_toe_neural_net/dense_7/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(²
4tic_tac_toe_neural_net/dense_8/MatMul/ReadVariableOpReadVariableOp=tic_tac_toe_neural_net_dense_8_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0Ó
%tic_tac_toe_neural_net/dense_8/MatMulMatMul2tic_tac_toe_neural_net/dropout/Identity_6:output:0<tic_tac_toe_neural_net/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
5tic_tac_toe_neural_net/dense_8/BiasAdd/ReadVariableOpReadVariableOp>tic_tac_toe_neural_net_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ó
&tic_tac_toe_neural_net/dense_8/BiasAddBiasAdd/tic_tac_toe_neural_net/dense_8/MatMul:product:0=tic_tac_toe_neural_net/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#tic_tac_toe_neural_net/dense_8/TanhTanh/tic_tac_toe_neural_net/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity0tic_tac_toe_neural_net/dense_5/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	x

Identity_1Identity'tic_tac_toe_neural_net/dense_8/Tanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
NoOpNoOpK^tic_tac_toe_neural_net/batch_normalization/FusedBatchNormV3/ReadVariableOpM^tic_tac_toe_neural_net/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:^tic_tac_toe_neural_net/batch_normalization/ReadVariableOp<^tic_tac_toe_neural_net/batch_normalization/ReadVariableOp_1M^tic_tac_toe_neural_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOpO^tic_tac_toe_neural_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1<^tic_tac_toe_neural_net/batch_normalization_1/ReadVariableOp>^tic_tac_toe_neural_net/batch_normalization_1/ReadVariableOp_1M^tic_tac_toe_neural_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOpO^tic_tac_toe_neural_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1<^tic_tac_toe_neural_net/batch_normalization_2/ReadVariableOp>^tic_tac_toe_neural_net/batch_normalization_2/ReadVariableOp_1M^tic_tac_toe_neural_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOpO^tic_tac_toe_neural_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1<^tic_tac_toe_neural_net/batch_normalization_3/ReadVariableOp>^tic_tac_toe_neural_net/batch_normalization_3/ReadVariableOp_1F^tic_tac_toe_neural_net/batch_normalization_4/batchnorm/ReadVariableOpH^tic_tac_toe_neural_net/batch_normalization_4/batchnorm/ReadVariableOp_1H^tic_tac_toe_neural_net/batch_normalization_4/batchnorm/ReadVariableOp_2J^tic_tac_toe_neural_net/batch_normalization_4/batchnorm/mul/ReadVariableOpF^tic_tac_toe_neural_net/batch_normalization_5/batchnorm/ReadVariableOpH^tic_tac_toe_neural_net/batch_normalization_5/batchnorm/ReadVariableOp_1H^tic_tac_toe_neural_net/batch_normalization_5/batchnorm/ReadVariableOp_2J^tic_tac_toe_neural_net/batch_normalization_5/batchnorm/mul/ReadVariableOp5^tic_tac_toe_neural_net/conv2d/BiasAdd/ReadVariableOp4^tic_tac_toe_neural_net/conv2d/Conv2D/ReadVariableOp7^tic_tac_toe_neural_net/conv2d_1/BiasAdd/ReadVariableOp6^tic_tac_toe_neural_net/conv2d_1/Conv2D/ReadVariableOp7^tic_tac_toe_neural_net/conv2d_2/BiasAdd/ReadVariableOp6^tic_tac_toe_neural_net/conv2d_2/Conv2D/ReadVariableOp7^tic_tac_toe_neural_net/conv2d_3/BiasAdd/ReadVariableOp6^tic_tac_toe_neural_net/conv2d_3/Conv2D/ReadVariableOp4^tic_tac_toe_neural_net/dense/BiasAdd/ReadVariableOp3^tic_tac_toe_neural_net/dense/MatMul/ReadVariableOp6^tic_tac_toe_neural_net/dense_1/BiasAdd/ReadVariableOp5^tic_tac_toe_neural_net/dense_1/MatMul/ReadVariableOp6^tic_tac_toe_neural_net/dense_2/BiasAdd/ReadVariableOp5^tic_tac_toe_neural_net/dense_2/MatMul/ReadVariableOp6^tic_tac_toe_neural_net/dense_3/BiasAdd/ReadVariableOp5^tic_tac_toe_neural_net/dense_3/MatMul/ReadVariableOp6^tic_tac_toe_neural_net/dense_4/BiasAdd/ReadVariableOp5^tic_tac_toe_neural_net/dense_4/MatMul/ReadVariableOp6^tic_tac_toe_neural_net/dense_5/BiasAdd/ReadVariableOp5^tic_tac_toe_neural_net/dense_5/MatMul/ReadVariableOp6^tic_tac_toe_neural_net/dense_6/BiasAdd/ReadVariableOp5^tic_tac_toe_neural_net/dense_6/MatMul/ReadVariableOp6^tic_tac_toe_neural_net/dense_7/BiasAdd/ReadVariableOp5^tic_tac_toe_neural_net/dense_7/MatMul/ReadVariableOp6^tic_tac_toe_neural_net/dense_8/BiasAdd/ReadVariableOp5^tic_tac_toe_neural_net/dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
Jtic_tac_toe_neural_net/batch_normalization/FusedBatchNormV3/ReadVariableOpJtic_tac_toe_neural_net/batch_normalization/FusedBatchNormV3/ReadVariableOp2
Ltic_tac_toe_neural_net/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ltic_tac_toe_neural_net/batch_normalization/FusedBatchNormV3/ReadVariableOp_12v
9tic_tac_toe_neural_net/batch_normalization/ReadVariableOp9tic_tac_toe_neural_net/batch_normalization/ReadVariableOp2z
;tic_tac_toe_neural_net/batch_normalization/ReadVariableOp_1;tic_tac_toe_neural_net/batch_normalization/ReadVariableOp_12
Ltic_tac_toe_neural_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOpLtic_tac_toe_neural_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2 
Ntic_tac_toe_neural_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ntic_tac_toe_neural_net/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12z
;tic_tac_toe_neural_net/batch_normalization_1/ReadVariableOp;tic_tac_toe_neural_net/batch_normalization_1/ReadVariableOp2~
=tic_tac_toe_neural_net/batch_normalization_1/ReadVariableOp_1=tic_tac_toe_neural_net/batch_normalization_1/ReadVariableOp_12
Ltic_tac_toe_neural_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOpLtic_tac_toe_neural_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2 
Ntic_tac_toe_neural_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ntic_tac_toe_neural_net/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12z
;tic_tac_toe_neural_net/batch_normalization_2/ReadVariableOp;tic_tac_toe_neural_net/batch_normalization_2/ReadVariableOp2~
=tic_tac_toe_neural_net/batch_normalization_2/ReadVariableOp_1=tic_tac_toe_neural_net/batch_normalization_2/ReadVariableOp_12
Ltic_tac_toe_neural_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOpLtic_tac_toe_neural_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2 
Ntic_tac_toe_neural_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ntic_tac_toe_neural_net/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12z
;tic_tac_toe_neural_net/batch_normalization_3/ReadVariableOp;tic_tac_toe_neural_net/batch_normalization_3/ReadVariableOp2~
=tic_tac_toe_neural_net/batch_normalization_3/ReadVariableOp_1=tic_tac_toe_neural_net/batch_normalization_3/ReadVariableOp_12
Etic_tac_toe_neural_net/batch_normalization_4/batchnorm/ReadVariableOpEtic_tac_toe_neural_net/batch_normalization_4/batchnorm/ReadVariableOp2
Gtic_tac_toe_neural_net/batch_normalization_4/batchnorm/ReadVariableOp_1Gtic_tac_toe_neural_net/batch_normalization_4/batchnorm/ReadVariableOp_12
Gtic_tac_toe_neural_net/batch_normalization_4/batchnorm/ReadVariableOp_2Gtic_tac_toe_neural_net/batch_normalization_4/batchnorm/ReadVariableOp_22
Itic_tac_toe_neural_net/batch_normalization_4/batchnorm/mul/ReadVariableOpItic_tac_toe_neural_net/batch_normalization_4/batchnorm/mul/ReadVariableOp2
Etic_tac_toe_neural_net/batch_normalization_5/batchnorm/ReadVariableOpEtic_tac_toe_neural_net/batch_normalization_5/batchnorm/ReadVariableOp2
Gtic_tac_toe_neural_net/batch_normalization_5/batchnorm/ReadVariableOp_1Gtic_tac_toe_neural_net/batch_normalization_5/batchnorm/ReadVariableOp_12
Gtic_tac_toe_neural_net/batch_normalization_5/batchnorm/ReadVariableOp_2Gtic_tac_toe_neural_net/batch_normalization_5/batchnorm/ReadVariableOp_22
Itic_tac_toe_neural_net/batch_normalization_5/batchnorm/mul/ReadVariableOpItic_tac_toe_neural_net/batch_normalization_5/batchnorm/mul/ReadVariableOp2l
4tic_tac_toe_neural_net/conv2d/BiasAdd/ReadVariableOp4tic_tac_toe_neural_net/conv2d/BiasAdd/ReadVariableOp2j
3tic_tac_toe_neural_net/conv2d/Conv2D/ReadVariableOp3tic_tac_toe_neural_net/conv2d/Conv2D/ReadVariableOp2p
6tic_tac_toe_neural_net/conv2d_1/BiasAdd/ReadVariableOp6tic_tac_toe_neural_net/conv2d_1/BiasAdd/ReadVariableOp2n
5tic_tac_toe_neural_net/conv2d_1/Conv2D/ReadVariableOp5tic_tac_toe_neural_net/conv2d_1/Conv2D/ReadVariableOp2p
6tic_tac_toe_neural_net/conv2d_2/BiasAdd/ReadVariableOp6tic_tac_toe_neural_net/conv2d_2/BiasAdd/ReadVariableOp2n
5tic_tac_toe_neural_net/conv2d_2/Conv2D/ReadVariableOp5tic_tac_toe_neural_net/conv2d_2/Conv2D/ReadVariableOp2p
6tic_tac_toe_neural_net/conv2d_3/BiasAdd/ReadVariableOp6tic_tac_toe_neural_net/conv2d_3/BiasAdd/ReadVariableOp2n
5tic_tac_toe_neural_net/conv2d_3/Conv2D/ReadVariableOp5tic_tac_toe_neural_net/conv2d_3/Conv2D/ReadVariableOp2j
3tic_tac_toe_neural_net/dense/BiasAdd/ReadVariableOp3tic_tac_toe_neural_net/dense/BiasAdd/ReadVariableOp2h
2tic_tac_toe_neural_net/dense/MatMul/ReadVariableOp2tic_tac_toe_neural_net/dense/MatMul/ReadVariableOp2n
5tic_tac_toe_neural_net/dense_1/BiasAdd/ReadVariableOp5tic_tac_toe_neural_net/dense_1/BiasAdd/ReadVariableOp2l
4tic_tac_toe_neural_net/dense_1/MatMul/ReadVariableOp4tic_tac_toe_neural_net/dense_1/MatMul/ReadVariableOp2n
5tic_tac_toe_neural_net/dense_2/BiasAdd/ReadVariableOp5tic_tac_toe_neural_net/dense_2/BiasAdd/ReadVariableOp2l
4tic_tac_toe_neural_net/dense_2/MatMul/ReadVariableOp4tic_tac_toe_neural_net/dense_2/MatMul/ReadVariableOp2n
5tic_tac_toe_neural_net/dense_3/BiasAdd/ReadVariableOp5tic_tac_toe_neural_net/dense_3/BiasAdd/ReadVariableOp2l
4tic_tac_toe_neural_net/dense_3/MatMul/ReadVariableOp4tic_tac_toe_neural_net/dense_3/MatMul/ReadVariableOp2n
5tic_tac_toe_neural_net/dense_4/BiasAdd/ReadVariableOp5tic_tac_toe_neural_net/dense_4/BiasAdd/ReadVariableOp2l
4tic_tac_toe_neural_net/dense_4/MatMul/ReadVariableOp4tic_tac_toe_neural_net/dense_4/MatMul/ReadVariableOp2n
5tic_tac_toe_neural_net/dense_5/BiasAdd/ReadVariableOp5tic_tac_toe_neural_net/dense_5/BiasAdd/ReadVariableOp2l
4tic_tac_toe_neural_net/dense_5/MatMul/ReadVariableOp4tic_tac_toe_neural_net/dense_5/MatMul/ReadVariableOp2n
5tic_tac_toe_neural_net/dense_6/BiasAdd/ReadVariableOp5tic_tac_toe_neural_net/dense_6/BiasAdd/ReadVariableOp2l
4tic_tac_toe_neural_net/dense_6/MatMul/ReadVariableOp4tic_tac_toe_neural_net/dense_6/MatMul/ReadVariableOp2n
5tic_tac_toe_neural_net/dense_7/BiasAdd/ReadVariableOp5tic_tac_toe_neural_net/dense_7/BiasAdd/ReadVariableOp2l
4tic_tac_toe_neural_net/dense_7/MatMul/ReadVariableOp4tic_tac_toe_neural_net/dense_7/MatMul/ReadVariableOp2n
5tic_tac_toe_neural_net/dense_8/BiasAdd/ReadVariableOp5tic_tac_toe_neural_net/dense_8/BiasAdd/ReadVariableOp2l
4tic_tac_toe_neural_net/dense_8/MatMul/ReadVariableOp4tic_tac_toe_neural_net/dense_8/MatMul/ReadVariableOp:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
£
F
*__inference_dropout_layer_call_fn_41075030

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072409a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
×
8__inference_batch_normalization_3_layer_call_fn_41074740

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_41072052
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´


F__inference_conv2d_2_layer_call_and_return_conditional_losses_41074633

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ã
T__inference_tic_tac_toe_neural_net_layer_call_and_return_conditional_losses_41073528
input_1*
conv2d_41073393:
conv2d_41073395:	+
batch_normalization_41073398:	+
batch_normalization_41073400:	+
batch_normalization_41073402:	+
batch_normalization_41073404:	-
conv2d_1_41073408: 
conv2d_1_41073410:	-
batch_normalization_1_41073413:	-
batch_normalization_1_41073415:	-
batch_normalization_1_41073417:	-
batch_normalization_1_41073419:	-
conv2d_2_41073423: 
conv2d_2_41073425:	-
batch_normalization_2_41073428:	-
batch_normalization_2_41073430:	-
batch_normalization_2_41073432:	-
batch_normalization_2_41073434:	-
conv2d_3_41073438: 
conv2d_3_41073440:	-
batch_normalization_3_41073443:	-
batch_normalization_3_41073445:	-
batch_normalization_3_41073447:	-
batch_normalization_3_41073449:	"
dense_41073454:

dense_41073456:	-
batch_normalization_4_41073459:	-
batch_normalization_4_41073461:	-
batch_normalization_4_41073463:	-
batch_normalization_4_41073465:	$
dense_1_41073470:

dense_1_41073472:	-
batch_normalization_5_41073475:	-
batch_normalization_5_41073477:	-
batch_normalization_5_41073479:	-
batch_normalization_5_41073481:	#
dense_2_41073486:	2
dense_2_41073488:2"
dense_3_41073492:2
dense_3_41073494:"
dense_4_41073498:
dense_4_41073500:"
dense_5_41073504:	
dense_5_41073506:	#
dense_6_41073509:	2
dense_6_41073511:2"
dense_7_41073515:2(
dense_7_41073517:("
dense_8_41073521:(
dense_8_41073523:
identity

identity_1¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢dense_8/StatefulPartitionedCallø
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_41073393conv2d_41073395*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_41072244
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_41073398batch_normalization_41073400batch_normalization_41073402batch_normalization_41073404*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_41071829}
ReluRelu4batch_normalization/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0conv2d_1_41073408conv2d_1_41073410*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_41072270
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_41073413batch_normalization_1_41073415batch_normalization_1_41073417batch_normalization_1_41073419*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_41071893
Relu_1Relu6batch_normalization_1/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0conv2d_2_41073423conv2d_2_41073425*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_41072296
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_41073428batch_normalization_2_41073430batch_normalization_2_41073432batch_normalization_2_41073434*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_41071957
Relu_2Relu6batch_normalization_2/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0conv2d_3_41073438conv2d_3_41073440*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_41072322
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_41073443batch_normalization_3_41073445batch_normalization_3_41073447batch_normalization_3_41073449*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_41072021
Relu_3Relu6batch_normalization_3/StatefulPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
flatten/PartitionedCallPartitionedCallRelu_3:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_41072344
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_41073454dense_41073456*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_41072356
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_4_41073459batch_normalization_4_41073461batch_normalization_4_41073463batch_normalization_4_41073465*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_41072087y
Relu_4Relu6batch_normalization_4/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
dropout/PartitionedCallPartitionedCallRelu_4:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072377
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_41073470dense_1_41073472*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_41072389
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_5_41073475batch_normalization_5_41073477batch_normalization_5_41073479batch_normalization_5_41073481*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_41072169y
Relu_5Relu6batch_normalization_5/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
dropout/PartitionedCall_1PartitionedCallRelu_5:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072409
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout/PartitionedCall_1:output:0dense_2_41073486dense_2_41073488*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_41072422Ü
dropout/PartitionedCall_2PartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072432
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout/PartitionedCall_2:output:0dense_3_41073492dense_3_41073494*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_41072445Ü
dropout/PartitionedCall_3PartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072455
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout/PartitionedCall_3:output:0dense_4_41073498dense_4_41073500*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_41072468Ü
dropout/PartitionedCall_4PartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072478
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout/PartitionedCall_4:output:0dense_5_41073504dense_5_41073506*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_41072491
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout/PartitionedCall_1:output:0dense_6_41073509dense_6_41073511*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_41072508Ü
dropout/PartitionedCall_5PartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072432
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout/PartitionedCall_5:output:0dense_7_41073515dense_7_41073517*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_41072526Ü
dropout/PartitionedCall_6PartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072536
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout/PartitionedCall_6:output:0dense_8_41073521dense_8_41073523*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_41072549w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	y

Identity_1Identity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ä

*__inference_dense_4_layer_call_fn_41075196

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_41072468o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ö
E__inference_dense_4_layer_call_and_return_conditional_losses_41072468

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´


F__inference_conv2d_1_layer_call_and_return_conditional_losses_41072270

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

F
*__inference_dropout_layer_call_fn_41075010

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072478`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
c
*__inference_dropout_layer_call_fn_41075015

inputs
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072766o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
c
E__inference_dropout_layer_call_and_return_conditional_losses_41075070

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

*__inference_dense_3_layer_call_fn_41075176

inputs
unknown:2
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_41072445o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
û	
d
E__inference_dropout_layer_call_and_return_conditional_losses_41075147

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
ïQ
!__inference__traced_save_41075719
file_prefixC
?savev2_tic_tac_toe_neural_net_conv2d_kernel_read_readvariableopA
=savev2_tic_tac_toe_neural_net_conv2d_bias_read_readvariableopO
Ksavev2_tic_tac_toe_neural_net_batch_normalization_gamma_read_readvariableopN
Jsavev2_tic_tac_toe_neural_net_batch_normalization_beta_read_readvariableopU
Qsavev2_tic_tac_toe_neural_net_batch_normalization_moving_mean_read_readvariableopY
Usavev2_tic_tac_toe_neural_net_batch_normalization_moving_variance_read_readvariableopE
Asavev2_tic_tac_toe_neural_net_conv2d_1_kernel_read_readvariableopC
?savev2_tic_tac_toe_neural_net_conv2d_1_bias_read_readvariableopQ
Msavev2_tic_tac_toe_neural_net_batch_normalization_1_gamma_read_readvariableopP
Lsavev2_tic_tac_toe_neural_net_batch_normalization_1_beta_read_readvariableopW
Ssavev2_tic_tac_toe_neural_net_batch_normalization_1_moving_mean_read_readvariableop[
Wsavev2_tic_tac_toe_neural_net_batch_normalization_1_moving_variance_read_readvariableopE
Asavev2_tic_tac_toe_neural_net_conv2d_2_kernel_read_readvariableopC
?savev2_tic_tac_toe_neural_net_conv2d_2_bias_read_readvariableopQ
Msavev2_tic_tac_toe_neural_net_batch_normalization_2_gamma_read_readvariableopP
Lsavev2_tic_tac_toe_neural_net_batch_normalization_2_beta_read_readvariableopW
Ssavev2_tic_tac_toe_neural_net_batch_normalization_2_moving_mean_read_readvariableop[
Wsavev2_tic_tac_toe_neural_net_batch_normalization_2_moving_variance_read_readvariableopE
Asavev2_tic_tac_toe_neural_net_conv2d_3_kernel_read_readvariableopC
?savev2_tic_tac_toe_neural_net_conv2d_3_bias_read_readvariableopQ
Msavev2_tic_tac_toe_neural_net_batch_normalization_3_gamma_read_readvariableopP
Lsavev2_tic_tac_toe_neural_net_batch_normalization_3_beta_read_readvariableopW
Ssavev2_tic_tac_toe_neural_net_batch_normalization_3_moving_mean_read_readvariableop[
Wsavev2_tic_tac_toe_neural_net_batch_normalization_3_moving_variance_read_readvariableopB
>savev2_tic_tac_toe_neural_net_dense_kernel_read_readvariableop@
<savev2_tic_tac_toe_neural_net_dense_bias_read_readvariableopQ
Msavev2_tic_tac_toe_neural_net_batch_normalization_4_gamma_read_readvariableopP
Lsavev2_tic_tac_toe_neural_net_batch_normalization_4_beta_read_readvariableopW
Ssavev2_tic_tac_toe_neural_net_batch_normalization_4_moving_mean_read_readvariableop[
Wsavev2_tic_tac_toe_neural_net_batch_normalization_4_moving_variance_read_readvariableopD
@savev2_tic_tac_toe_neural_net_dense_1_kernel_read_readvariableopB
>savev2_tic_tac_toe_neural_net_dense_1_bias_read_readvariableopQ
Msavev2_tic_tac_toe_neural_net_batch_normalization_5_gamma_read_readvariableopP
Lsavev2_tic_tac_toe_neural_net_batch_normalization_5_beta_read_readvariableopW
Ssavev2_tic_tac_toe_neural_net_batch_normalization_5_moving_mean_read_readvariableop[
Wsavev2_tic_tac_toe_neural_net_batch_normalization_5_moving_variance_read_readvariableopD
@savev2_tic_tac_toe_neural_net_dense_2_kernel_read_readvariableopB
>savev2_tic_tac_toe_neural_net_dense_2_bias_read_readvariableopD
@savev2_tic_tac_toe_neural_net_dense_3_kernel_read_readvariableopB
>savev2_tic_tac_toe_neural_net_dense_3_bias_read_readvariableopD
@savev2_tic_tac_toe_neural_net_dense_4_kernel_read_readvariableopB
>savev2_tic_tac_toe_neural_net_dense_4_bias_read_readvariableopD
@savev2_tic_tac_toe_neural_net_dense_5_kernel_read_readvariableopB
>savev2_tic_tac_toe_neural_net_dense_5_bias_read_readvariableopD
@savev2_tic_tac_toe_neural_net_dense_6_kernel_read_readvariableopB
>savev2_tic_tac_toe_neural_net_dense_6_bias_read_readvariableopD
@savev2_tic_tac_toe_neural_net_dense_7_kernel_read_readvariableopB
>savev2_tic_tac_toe_neural_net_dense_7_bias_read_readvariableopD
@savev2_tic_tac_toe_neural_net_dense_8_kernel_read_readvariableopB
>savev2_tic_tac_toe_neural_net_dense_8_bias_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableopJ
Fsavev2_adam_tic_tac_toe_neural_net_conv2d_kernel_m_read_readvariableopH
Dsavev2_adam_tic_tac_toe_neural_net_conv2d_bias_m_read_readvariableopV
Rsavev2_adam_tic_tac_toe_neural_net_batch_normalization_gamma_m_read_readvariableopU
Qsavev2_adam_tic_tac_toe_neural_net_batch_normalization_beta_m_read_readvariableopL
Hsavev2_adam_tic_tac_toe_neural_net_conv2d_1_kernel_m_read_readvariableopJ
Fsavev2_adam_tic_tac_toe_neural_net_conv2d_1_bias_m_read_readvariableopX
Tsavev2_adam_tic_tac_toe_neural_net_batch_normalization_1_gamma_m_read_readvariableopW
Ssavev2_adam_tic_tac_toe_neural_net_batch_normalization_1_beta_m_read_readvariableopL
Hsavev2_adam_tic_tac_toe_neural_net_conv2d_2_kernel_m_read_readvariableopJ
Fsavev2_adam_tic_tac_toe_neural_net_conv2d_2_bias_m_read_readvariableopX
Tsavev2_adam_tic_tac_toe_neural_net_batch_normalization_2_gamma_m_read_readvariableopW
Ssavev2_adam_tic_tac_toe_neural_net_batch_normalization_2_beta_m_read_readvariableopL
Hsavev2_adam_tic_tac_toe_neural_net_conv2d_3_kernel_m_read_readvariableopJ
Fsavev2_adam_tic_tac_toe_neural_net_conv2d_3_bias_m_read_readvariableopX
Tsavev2_adam_tic_tac_toe_neural_net_batch_normalization_3_gamma_m_read_readvariableopW
Ssavev2_adam_tic_tac_toe_neural_net_batch_normalization_3_beta_m_read_readvariableopI
Esavev2_adam_tic_tac_toe_neural_net_dense_kernel_m_read_readvariableopG
Csavev2_adam_tic_tac_toe_neural_net_dense_bias_m_read_readvariableopX
Tsavev2_adam_tic_tac_toe_neural_net_batch_normalization_4_gamma_m_read_readvariableopW
Ssavev2_adam_tic_tac_toe_neural_net_batch_normalization_4_beta_m_read_readvariableopK
Gsavev2_adam_tic_tac_toe_neural_net_dense_1_kernel_m_read_readvariableopI
Esavev2_adam_tic_tac_toe_neural_net_dense_1_bias_m_read_readvariableopX
Tsavev2_adam_tic_tac_toe_neural_net_batch_normalization_5_gamma_m_read_readvariableopW
Ssavev2_adam_tic_tac_toe_neural_net_batch_normalization_5_beta_m_read_readvariableopK
Gsavev2_adam_tic_tac_toe_neural_net_dense_2_kernel_m_read_readvariableopI
Esavev2_adam_tic_tac_toe_neural_net_dense_2_bias_m_read_readvariableopK
Gsavev2_adam_tic_tac_toe_neural_net_dense_3_kernel_m_read_readvariableopI
Esavev2_adam_tic_tac_toe_neural_net_dense_3_bias_m_read_readvariableopK
Gsavev2_adam_tic_tac_toe_neural_net_dense_4_kernel_m_read_readvariableopI
Esavev2_adam_tic_tac_toe_neural_net_dense_4_bias_m_read_readvariableopK
Gsavev2_adam_tic_tac_toe_neural_net_dense_5_kernel_m_read_readvariableopI
Esavev2_adam_tic_tac_toe_neural_net_dense_5_bias_m_read_readvariableopK
Gsavev2_adam_tic_tac_toe_neural_net_dense_6_kernel_m_read_readvariableopI
Esavev2_adam_tic_tac_toe_neural_net_dense_6_bias_m_read_readvariableopK
Gsavev2_adam_tic_tac_toe_neural_net_dense_7_kernel_m_read_readvariableopI
Esavev2_adam_tic_tac_toe_neural_net_dense_7_bias_m_read_readvariableopK
Gsavev2_adam_tic_tac_toe_neural_net_dense_8_kernel_m_read_readvariableopI
Esavev2_adam_tic_tac_toe_neural_net_dense_8_bias_m_read_readvariableopJ
Fsavev2_adam_tic_tac_toe_neural_net_conv2d_kernel_v_read_readvariableopH
Dsavev2_adam_tic_tac_toe_neural_net_conv2d_bias_v_read_readvariableopV
Rsavev2_adam_tic_tac_toe_neural_net_batch_normalization_gamma_v_read_readvariableopU
Qsavev2_adam_tic_tac_toe_neural_net_batch_normalization_beta_v_read_readvariableopL
Hsavev2_adam_tic_tac_toe_neural_net_conv2d_1_kernel_v_read_readvariableopJ
Fsavev2_adam_tic_tac_toe_neural_net_conv2d_1_bias_v_read_readvariableopX
Tsavev2_adam_tic_tac_toe_neural_net_batch_normalization_1_gamma_v_read_readvariableopW
Ssavev2_adam_tic_tac_toe_neural_net_batch_normalization_1_beta_v_read_readvariableopL
Hsavev2_adam_tic_tac_toe_neural_net_conv2d_2_kernel_v_read_readvariableopJ
Fsavev2_adam_tic_tac_toe_neural_net_conv2d_2_bias_v_read_readvariableopX
Tsavev2_adam_tic_tac_toe_neural_net_batch_normalization_2_gamma_v_read_readvariableopW
Ssavev2_adam_tic_tac_toe_neural_net_batch_normalization_2_beta_v_read_readvariableopL
Hsavev2_adam_tic_tac_toe_neural_net_conv2d_3_kernel_v_read_readvariableopJ
Fsavev2_adam_tic_tac_toe_neural_net_conv2d_3_bias_v_read_readvariableopX
Tsavev2_adam_tic_tac_toe_neural_net_batch_normalization_3_gamma_v_read_readvariableopW
Ssavev2_adam_tic_tac_toe_neural_net_batch_normalization_3_beta_v_read_readvariableopI
Esavev2_adam_tic_tac_toe_neural_net_dense_kernel_v_read_readvariableopG
Csavev2_adam_tic_tac_toe_neural_net_dense_bias_v_read_readvariableopX
Tsavev2_adam_tic_tac_toe_neural_net_batch_normalization_4_gamma_v_read_readvariableopW
Ssavev2_adam_tic_tac_toe_neural_net_batch_normalization_4_beta_v_read_readvariableopK
Gsavev2_adam_tic_tac_toe_neural_net_dense_1_kernel_v_read_readvariableopI
Esavev2_adam_tic_tac_toe_neural_net_dense_1_bias_v_read_readvariableopX
Tsavev2_adam_tic_tac_toe_neural_net_batch_normalization_5_gamma_v_read_readvariableopW
Ssavev2_adam_tic_tac_toe_neural_net_batch_normalization_5_beta_v_read_readvariableopK
Gsavev2_adam_tic_tac_toe_neural_net_dense_2_kernel_v_read_readvariableopI
Esavev2_adam_tic_tac_toe_neural_net_dense_2_bias_v_read_readvariableopK
Gsavev2_adam_tic_tac_toe_neural_net_dense_3_kernel_v_read_readvariableopI
Esavev2_adam_tic_tac_toe_neural_net_dense_3_bias_v_read_readvariableopK
Gsavev2_adam_tic_tac_toe_neural_net_dense_4_kernel_v_read_readvariableopI
Esavev2_adam_tic_tac_toe_neural_net_dense_4_bias_v_read_readvariableopK
Gsavev2_adam_tic_tac_toe_neural_net_dense_5_kernel_v_read_readvariableopI
Esavev2_adam_tic_tac_toe_neural_net_dense_5_bias_v_read_readvariableopK
Gsavev2_adam_tic_tac_toe_neural_net_dense_6_kernel_v_read_readvariableopI
Esavev2_adam_tic_tac_toe_neural_net_dense_6_bias_v_read_readvariableopK
Gsavev2_adam_tic_tac_toe_neural_net_dense_7_kernel_v_read_readvariableopI
Esavev2_adam_tic_tac_toe_neural_net_dense_7_bias_v_read_readvariableopK
Gsavev2_adam_tic_tac_toe_neural_net_dense_8_kernel_v_read_readvariableopI
Esavev2_adam_tic_tac_toe_neural_net_dense_8_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: =
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*¿<
valueµ<B²<B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/40/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/41/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/42/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/43/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/44/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/45/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/46/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/47/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/48/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/49/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/40/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/41/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/42/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/43/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/44/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/45/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/46/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/47/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/48/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/49/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*¨
valueBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B O
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0?savev2_tic_tac_toe_neural_net_conv2d_kernel_read_readvariableop=savev2_tic_tac_toe_neural_net_conv2d_bias_read_readvariableopKsavev2_tic_tac_toe_neural_net_batch_normalization_gamma_read_readvariableopJsavev2_tic_tac_toe_neural_net_batch_normalization_beta_read_readvariableopQsavev2_tic_tac_toe_neural_net_batch_normalization_moving_mean_read_readvariableopUsavev2_tic_tac_toe_neural_net_batch_normalization_moving_variance_read_readvariableopAsavev2_tic_tac_toe_neural_net_conv2d_1_kernel_read_readvariableop?savev2_tic_tac_toe_neural_net_conv2d_1_bias_read_readvariableopMsavev2_tic_tac_toe_neural_net_batch_normalization_1_gamma_read_readvariableopLsavev2_tic_tac_toe_neural_net_batch_normalization_1_beta_read_readvariableopSsavev2_tic_tac_toe_neural_net_batch_normalization_1_moving_mean_read_readvariableopWsavev2_tic_tac_toe_neural_net_batch_normalization_1_moving_variance_read_readvariableopAsavev2_tic_tac_toe_neural_net_conv2d_2_kernel_read_readvariableop?savev2_tic_tac_toe_neural_net_conv2d_2_bias_read_readvariableopMsavev2_tic_tac_toe_neural_net_batch_normalization_2_gamma_read_readvariableopLsavev2_tic_tac_toe_neural_net_batch_normalization_2_beta_read_readvariableopSsavev2_tic_tac_toe_neural_net_batch_normalization_2_moving_mean_read_readvariableopWsavev2_tic_tac_toe_neural_net_batch_normalization_2_moving_variance_read_readvariableopAsavev2_tic_tac_toe_neural_net_conv2d_3_kernel_read_readvariableop?savev2_tic_tac_toe_neural_net_conv2d_3_bias_read_readvariableopMsavev2_tic_tac_toe_neural_net_batch_normalization_3_gamma_read_readvariableopLsavev2_tic_tac_toe_neural_net_batch_normalization_3_beta_read_readvariableopSsavev2_tic_tac_toe_neural_net_batch_normalization_3_moving_mean_read_readvariableopWsavev2_tic_tac_toe_neural_net_batch_normalization_3_moving_variance_read_readvariableop>savev2_tic_tac_toe_neural_net_dense_kernel_read_readvariableop<savev2_tic_tac_toe_neural_net_dense_bias_read_readvariableopMsavev2_tic_tac_toe_neural_net_batch_normalization_4_gamma_read_readvariableopLsavev2_tic_tac_toe_neural_net_batch_normalization_4_beta_read_readvariableopSsavev2_tic_tac_toe_neural_net_batch_normalization_4_moving_mean_read_readvariableopWsavev2_tic_tac_toe_neural_net_batch_normalization_4_moving_variance_read_readvariableop@savev2_tic_tac_toe_neural_net_dense_1_kernel_read_readvariableop>savev2_tic_tac_toe_neural_net_dense_1_bias_read_readvariableopMsavev2_tic_tac_toe_neural_net_batch_normalization_5_gamma_read_readvariableopLsavev2_tic_tac_toe_neural_net_batch_normalization_5_beta_read_readvariableopSsavev2_tic_tac_toe_neural_net_batch_normalization_5_moving_mean_read_readvariableopWsavev2_tic_tac_toe_neural_net_batch_normalization_5_moving_variance_read_readvariableop@savev2_tic_tac_toe_neural_net_dense_2_kernel_read_readvariableop>savev2_tic_tac_toe_neural_net_dense_2_bias_read_readvariableop@savev2_tic_tac_toe_neural_net_dense_3_kernel_read_readvariableop>savev2_tic_tac_toe_neural_net_dense_3_bias_read_readvariableop@savev2_tic_tac_toe_neural_net_dense_4_kernel_read_readvariableop>savev2_tic_tac_toe_neural_net_dense_4_bias_read_readvariableop@savev2_tic_tac_toe_neural_net_dense_5_kernel_read_readvariableop>savev2_tic_tac_toe_neural_net_dense_5_bias_read_readvariableop@savev2_tic_tac_toe_neural_net_dense_6_kernel_read_readvariableop>savev2_tic_tac_toe_neural_net_dense_6_bias_read_readvariableop@savev2_tic_tac_toe_neural_net_dense_7_kernel_read_readvariableop>savev2_tic_tac_toe_neural_net_dense_7_bias_read_readvariableop@savev2_tic_tac_toe_neural_net_dense_8_kernel_read_readvariableop>savev2_tic_tac_toe_neural_net_dense_8_bias_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableopFsavev2_adam_tic_tac_toe_neural_net_conv2d_kernel_m_read_readvariableopDsavev2_adam_tic_tac_toe_neural_net_conv2d_bias_m_read_readvariableopRsavev2_adam_tic_tac_toe_neural_net_batch_normalization_gamma_m_read_readvariableopQsavev2_adam_tic_tac_toe_neural_net_batch_normalization_beta_m_read_readvariableopHsavev2_adam_tic_tac_toe_neural_net_conv2d_1_kernel_m_read_readvariableopFsavev2_adam_tic_tac_toe_neural_net_conv2d_1_bias_m_read_readvariableopTsavev2_adam_tic_tac_toe_neural_net_batch_normalization_1_gamma_m_read_readvariableopSsavev2_adam_tic_tac_toe_neural_net_batch_normalization_1_beta_m_read_readvariableopHsavev2_adam_tic_tac_toe_neural_net_conv2d_2_kernel_m_read_readvariableopFsavev2_adam_tic_tac_toe_neural_net_conv2d_2_bias_m_read_readvariableopTsavev2_adam_tic_tac_toe_neural_net_batch_normalization_2_gamma_m_read_readvariableopSsavev2_adam_tic_tac_toe_neural_net_batch_normalization_2_beta_m_read_readvariableopHsavev2_adam_tic_tac_toe_neural_net_conv2d_3_kernel_m_read_readvariableopFsavev2_adam_tic_tac_toe_neural_net_conv2d_3_bias_m_read_readvariableopTsavev2_adam_tic_tac_toe_neural_net_batch_normalization_3_gamma_m_read_readvariableopSsavev2_adam_tic_tac_toe_neural_net_batch_normalization_3_beta_m_read_readvariableopEsavev2_adam_tic_tac_toe_neural_net_dense_kernel_m_read_readvariableopCsavev2_adam_tic_tac_toe_neural_net_dense_bias_m_read_readvariableopTsavev2_adam_tic_tac_toe_neural_net_batch_normalization_4_gamma_m_read_readvariableopSsavev2_adam_tic_tac_toe_neural_net_batch_normalization_4_beta_m_read_readvariableopGsavev2_adam_tic_tac_toe_neural_net_dense_1_kernel_m_read_readvariableopEsavev2_adam_tic_tac_toe_neural_net_dense_1_bias_m_read_readvariableopTsavev2_adam_tic_tac_toe_neural_net_batch_normalization_5_gamma_m_read_readvariableopSsavev2_adam_tic_tac_toe_neural_net_batch_normalization_5_beta_m_read_readvariableopGsavev2_adam_tic_tac_toe_neural_net_dense_2_kernel_m_read_readvariableopEsavev2_adam_tic_tac_toe_neural_net_dense_2_bias_m_read_readvariableopGsavev2_adam_tic_tac_toe_neural_net_dense_3_kernel_m_read_readvariableopEsavev2_adam_tic_tac_toe_neural_net_dense_3_bias_m_read_readvariableopGsavev2_adam_tic_tac_toe_neural_net_dense_4_kernel_m_read_readvariableopEsavev2_adam_tic_tac_toe_neural_net_dense_4_bias_m_read_readvariableopGsavev2_adam_tic_tac_toe_neural_net_dense_5_kernel_m_read_readvariableopEsavev2_adam_tic_tac_toe_neural_net_dense_5_bias_m_read_readvariableopGsavev2_adam_tic_tac_toe_neural_net_dense_6_kernel_m_read_readvariableopEsavev2_adam_tic_tac_toe_neural_net_dense_6_bias_m_read_readvariableopGsavev2_adam_tic_tac_toe_neural_net_dense_7_kernel_m_read_readvariableopEsavev2_adam_tic_tac_toe_neural_net_dense_7_bias_m_read_readvariableopGsavev2_adam_tic_tac_toe_neural_net_dense_8_kernel_m_read_readvariableopEsavev2_adam_tic_tac_toe_neural_net_dense_8_bias_m_read_readvariableopFsavev2_adam_tic_tac_toe_neural_net_conv2d_kernel_v_read_readvariableopDsavev2_adam_tic_tac_toe_neural_net_conv2d_bias_v_read_readvariableopRsavev2_adam_tic_tac_toe_neural_net_batch_normalization_gamma_v_read_readvariableopQsavev2_adam_tic_tac_toe_neural_net_batch_normalization_beta_v_read_readvariableopHsavev2_adam_tic_tac_toe_neural_net_conv2d_1_kernel_v_read_readvariableopFsavev2_adam_tic_tac_toe_neural_net_conv2d_1_bias_v_read_readvariableopTsavev2_adam_tic_tac_toe_neural_net_batch_normalization_1_gamma_v_read_readvariableopSsavev2_adam_tic_tac_toe_neural_net_batch_normalization_1_beta_v_read_readvariableopHsavev2_adam_tic_tac_toe_neural_net_conv2d_2_kernel_v_read_readvariableopFsavev2_adam_tic_tac_toe_neural_net_conv2d_2_bias_v_read_readvariableopTsavev2_adam_tic_tac_toe_neural_net_batch_normalization_2_gamma_v_read_readvariableopSsavev2_adam_tic_tac_toe_neural_net_batch_normalization_2_beta_v_read_readvariableopHsavev2_adam_tic_tac_toe_neural_net_conv2d_3_kernel_v_read_readvariableopFsavev2_adam_tic_tac_toe_neural_net_conv2d_3_bias_v_read_readvariableopTsavev2_adam_tic_tac_toe_neural_net_batch_normalization_3_gamma_v_read_readvariableopSsavev2_adam_tic_tac_toe_neural_net_batch_normalization_3_beta_v_read_readvariableopEsavev2_adam_tic_tac_toe_neural_net_dense_kernel_v_read_readvariableopCsavev2_adam_tic_tac_toe_neural_net_dense_bias_v_read_readvariableopTsavev2_adam_tic_tac_toe_neural_net_batch_normalization_4_gamma_v_read_readvariableopSsavev2_adam_tic_tac_toe_neural_net_batch_normalization_4_beta_v_read_readvariableopGsavev2_adam_tic_tac_toe_neural_net_dense_1_kernel_v_read_readvariableopEsavev2_adam_tic_tac_toe_neural_net_dense_1_bias_v_read_readvariableopTsavev2_adam_tic_tac_toe_neural_net_batch_normalization_5_gamma_v_read_readvariableopSsavev2_adam_tic_tac_toe_neural_net_batch_normalization_5_beta_v_read_readvariableopGsavev2_adam_tic_tac_toe_neural_net_dense_2_kernel_v_read_readvariableopEsavev2_adam_tic_tac_toe_neural_net_dense_2_bias_v_read_readvariableopGsavev2_adam_tic_tac_toe_neural_net_dense_3_kernel_v_read_readvariableopEsavev2_adam_tic_tac_toe_neural_net_dense_3_bias_v_read_readvariableopGsavev2_adam_tic_tac_toe_neural_net_dense_4_kernel_v_read_readvariableopEsavev2_adam_tic_tac_toe_neural_net_dense_4_bias_v_read_readvariableopGsavev2_adam_tic_tac_toe_neural_net_dense_5_kernel_v_read_readvariableopEsavev2_adam_tic_tac_toe_neural_net_dense_5_bias_v_read_readvariableopGsavev2_adam_tic_tac_toe_neural_net_dense_6_kernel_v_read_readvariableopEsavev2_adam_tic_tac_toe_neural_net_dense_6_bias_v_read_readvariableopGsavev2_adam_tic_tac_toe_neural_net_dense_7_kernel_v_read_readvariableopEsavev2_adam_tic_tac_toe_neural_net_dense_7_bias_v_read_readvariableopGsavev2_adam_tic_tac_toe_neural_net_dense_8_kernel_v_read_readvariableopEsavev2_adam_tic_tac_toe_neural_net_dense_8_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*	
_input_shapesô
ñ: :::::::::::::::::::::::::
::::::
::::::	2:2:2::::	:	:	2:2:2(:(:(:: : : : : : : : : : :::::::::::::::::
::::
::::	2:2:2::::	:	:	2:2:2(:(:(::::::::::::::::::
::::
::::	2:2:2::::	:	:	2:2:2(:(:(:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!	

_output_shapes	
::!


_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::&"
 
_output_shapes
:
:! 

_output_shapes	
::!!

_output_shapes	
::!"

_output_shapes	
::!#

_output_shapes	
::!$

_output_shapes	
::%%!

_output_shapes
:	2: &

_output_shapes
:2:$' 

_output_shapes

:2: (

_output_shapes
::$) 

_output_shapes

:: *

_output_shapes
::$+ 

_output_shapes

:	: ,

_output_shapes
:	:%-!

_output_shapes
:	2: .

_output_shapes
:2:$/ 

_output_shapes

:2(: 0

_output_shapes
:(:$1 

_output_shapes

:(: 2

_output_shapes
::3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :-=)
'
_output_shapes
::!>

_output_shapes	
::!?

_output_shapes	
::!@

_output_shapes	
::.A*
(
_output_shapes
::!B

_output_shapes	
::!C

_output_shapes	
::!D

_output_shapes	
::.E*
(
_output_shapes
::!F

_output_shapes	
::!G

_output_shapes	
::!H

_output_shapes	
::.I*
(
_output_shapes
::!J

_output_shapes	
::!K

_output_shapes	
::!L

_output_shapes	
::&M"
 
_output_shapes
:
:!N

_output_shapes	
::!O

_output_shapes	
::!P

_output_shapes	
::&Q"
 
_output_shapes
:
:!R

_output_shapes	
::!S

_output_shapes	
::!T

_output_shapes	
::%U!

_output_shapes
:	2: V

_output_shapes
:2:$W 

_output_shapes

:2: X

_output_shapes
::$Y 

_output_shapes

:: Z

_output_shapes
::$[ 

_output_shapes

:	: \

_output_shapes
:	:%]!

_output_shapes
:	2: ^

_output_shapes
:2:$_ 

_output_shapes

:2(: `

_output_shapes
:(:$a 

_output_shapes

:(: b

_output_shapes
::-c)
'
_output_shapes
::!d

_output_shapes	
::!e

_output_shapes	
::!f

_output_shapes	
::.g*
(
_output_shapes
::!h

_output_shapes	
::!i

_output_shapes	
::!j

_output_shapes	
::.k*
(
_output_shapes
::!l

_output_shapes	
::!m

_output_shapes	
::!n

_output_shapes	
::.o*
(
_output_shapes
::!p

_output_shapes	
::!q

_output_shapes	
::!r

_output_shapes	
::&s"
 
_output_shapes
:
:!t

_output_shapes	
::!u

_output_shapes	
::!v

_output_shapes	
::&w"
 
_output_shapes
:
:!x

_output_shapes	
::!y

_output_shapes	
::!z

_output_shapes	
::%{!

_output_shapes
:	2: |

_output_shapes
:2:$} 

_output_shapes

:2: ~

_output_shapes
::$ 

_output_shapes

::!

_output_shapes
::% 

_output_shapes

:	:!

_output_shapes
:	:&!

_output_shapes
:	2:!

_output_shapes
:2:% 

_output_shapes

:2(:!

_output_shapes
:(:% 

_output_shapes

:(:!

_output_shapes
::

_output_shapes
: 
	
Õ
6__inference_batch_normalization_layer_call_fn_41074484

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_41071829
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
c
E__inference_dropout_layer_call_and_return_conditional_losses_41075050

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
c
E__inference_dropout_layer_call_and_return_conditional_losses_41072536

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ([

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ("!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
â
¶
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_41074951

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú

9__inference_tic_tac_toe_neural_net_layer_call_fn_41072662
input_1"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:


unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:


unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	

unknown_34:	

unknown_35:	2

unknown_36:2

unknown_37:2

unknown_38:

unknown_39:

unknown_40:

unknown_41:	

unknown_42:	

unknown_43:	2

unknown_44:2

unknown_45:2(

unknown_46:(

unknown_47:(

unknown_48:
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ	:ÿÿÿÿÿÿÿÿÿ*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_tic_tac_toe_neural_net_layer_call_and_return_conditional_losses_41072557o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
	
Õ
6__inference_batch_normalization_layer_call_fn_41074497

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_41071860
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
c
E__inference_dropout_layer_call_and_return_conditional_losses_41072377

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
×
8__inference_batch_normalization_3_layer_call_fn_41074727

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_41072021
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
c
E__inference_dropout_layer_call_and_return_conditional_losses_41075055

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
¢
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_41071893

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ö
E__inference_dense_8_layer_call_and_return_conditional_losses_41075287

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
µ


F__inference_conv2d_3_layer_call_and_return_conditional_losses_41072322

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ö
E__inference_dense_7_layer_call_and_return_conditional_losses_41072526

inputs0
matmul_readvariableop_resource:2(-
biasadd_readvariableop_resource:(
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
è

9__inference_tic_tac_toe_neural_net_layer_call_fn_41073882
x"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:


unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:


unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	

unknown_34:	

unknown_35:	2

unknown_36:2

unknown_37:2

unknown_38:

unknown_39:

unknown_40:

unknown_41:	

unknown_42:	

unknown_43:	2

unknown_44:2

unknown_45:2(

unknown_46:(

unknown_47:(

unknown_48:
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ	:ÿÿÿÿÿÿÿÿÿ*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./012*-
config_proto

CPU

GPU 2J 8 *]
fXRV
T__inference_tic_tac_toe_neural_net_layer_call_and_return_conditional_losses_41072557o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
õ
c
*__inference_dropout_layer_call_fn_41075035

inputs
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072840p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ
£
+__inference_conv2d_1_layer_call_fn_41074542

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_41072270x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Æ
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_41071924

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Æ
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_41074695

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç

(__inference_dense_layer_call_fn_41074796

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_41072356p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_41074515

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ä
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_41074533

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ


F__inference_conv2d_3_layer_call_and_return_conditional_losses_41074714

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó	
d
E__inference_dropout_layer_call_and_return_conditional_losses_41075111

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ
£
+__inference_conv2d_3_layer_call_fn_41074704

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_41072322x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
¶
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_41074852

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó	
d
E__inference_dropout_layer_call_and_return_conditional_losses_41072798

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
c
*__inference_dropout_layer_call_fn_41075025

inputs
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072798o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Æ
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_41074614

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
¶
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_41072169

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

F
*__inference_dropout_layer_call_fn_41074990

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072536`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
ªÀ
o
$__inference__traced_restore_41076137
file_prefixP
5assignvariableop_tic_tac_toe_neural_net_conv2d_kernel:D
5assignvariableop_1_tic_tac_toe_neural_net_conv2d_bias:	R
Cassignvariableop_2_tic_tac_toe_neural_net_batch_normalization_gamma:	Q
Bassignvariableop_3_tic_tac_toe_neural_net_batch_normalization_beta:	X
Iassignvariableop_4_tic_tac_toe_neural_net_batch_normalization_moving_mean:	\
Massignvariableop_5_tic_tac_toe_neural_net_batch_normalization_moving_variance:	U
9assignvariableop_6_tic_tac_toe_neural_net_conv2d_1_kernel:F
7assignvariableop_7_tic_tac_toe_neural_net_conv2d_1_bias:	T
Eassignvariableop_8_tic_tac_toe_neural_net_batch_normalization_1_gamma:	S
Dassignvariableop_9_tic_tac_toe_neural_net_batch_normalization_1_beta:	[
Lassignvariableop_10_tic_tac_toe_neural_net_batch_normalization_1_moving_mean:	_
Passignvariableop_11_tic_tac_toe_neural_net_batch_normalization_1_moving_variance:	V
:assignvariableop_12_tic_tac_toe_neural_net_conv2d_2_kernel:G
8assignvariableop_13_tic_tac_toe_neural_net_conv2d_2_bias:	U
Fassignvariableop_14_tic_tac_toe_neural_net_batch_normalization_2_gamma:	T
Eassignvariableop_15_tic_tac_toe_neural_net_batch_normalization_2_beta:	[
Lassignvariableop_16_tic_tac_toe_neural_net_batch_normalization_2_moving_mean:	_
Passignvariableop_17_tic_tac_toe_neural_net_batch_normalization_2_moving_variance:	V
:assignvariableop_18_tic_tac_toe_neural_net_conv2d_3_kernel:G
8assignvariableop_19_tic_tac_toe_neural_net_conv2d_3_bias:	U
Fassignvariableop_20_tic_tac_toe_neural_net_batch_normalization_3_gamma:	T
Eassignvariableop_21_tic_tac_toe_neural_net_batch_normalization_3_beta:	[
Lassignvariableop_22_tic_tac_toe_neural_net_batch_normalization_3_moving_mean:	_
Passignvariableop_23_tic_tac_toe_neural_net_batch_normalization_3_moving_variance:	K
7assignvariableop_24_tic_tac_toe_neural_net_dense_kernel:
D
5assignvariableop_25_tic_tac_toe_neural_net_dense_bias:	U
Fassignvariableop_26_tic_tac_toe_neural_net_batch_normalization_4_gamma:	T
Eassignvariableop_27_tic_tac_toe_neural_net_batch_normalization_4_beta:	[
Lassignvariableop_28_tic_tac_toe_neural_net_batch_normalization_4_moving_mean:	_
Passignvariableop_29_tic_tac_toe_neural_net_batch_normalization_4_moving_variance:	M
9assignvariableop_30_tic_tac_toe_neural_net_dense_1_kernel:
F
7assignvariableop_31_tic_tac_toe_neural_net_dense_1_bias:	U
Fassignvariableop_32_tic_tac_toe_neural_net_batch_normalization_5_gamma:	T
Eassignvariableop_33_tic_tac_toe_neural_net_batch_normalization_5_beta:	[
Lassignvariableop_34_tic_tac_toe_neural_net_batch_normalization_5_moving_mean:	_
Passignvariableop_35_tic_tac_toe_neural_net_batch_normalization_5_moving_variance:	L
9assignvariableop_36_tic_tac_toe_neural_net_dense_2_kernel:	2E
7assignvariableop_37_tic_tac_toe_neural_net_dense_2_bias:2K
9assignvariableop_38_tic_tac_toe_neural_net_dense_3_kernel:2E
7assignvariableop_39_tic_tac_toe_neural_net_dense_3_bias:K
9assignvariableop_40_tic_tac_toe_neural_net_dense_4_kernel:E
7assignvariableop_41_tic_tac_toe_neural_net_dense_4_bias:K
9assignvariableop_42_tic_tac_toe_neural_net_dense_5_kernel:	E
7assignvariableop_43_tic_tac_toe_neural_net_dense_5_bias:	L
9assignvariableop_44_tic_tac_toe_neural_net_dense_6_kernel:	2E
7assignvariableop_45_tic_tac_toe_neural_net_dense_6_bias:2K
9assignvariableop_46_tic_tac_toe_neural_net_dense_7_kernel:2(E
7assignvariableop_47_tic_tac_toe_neural_net_dense_7_bias:(K
9assignvariableop_48_tic_tac_toe_neural_net_dense_8_kernel:(E
7assignvariableop_49_tic_tac_toe_neural_net_dense_8_bias:%
assignvariableop_50_total_2: %
assignvariableop_51_count_2: %
assignvariableop_52_total_1: %
assignvariableop_53_count_1: #
assignvariableop_54_total: #
assignvariableop_55_count: '
assignvariableop_56_adam_iter:	 )
assignvariableop_57_adam_beta_1: )
assignvariableop_58_adam_beta_2: (
assignvariableop_59_adam_decay: Z
?assignvariableop_60_adam_tic_tac_toe_neural_net_conv2d_kernel_m:L
=assignvariableop_61_adam_tic_tac_toe_neural_net_conv2d_bias_m:	Z
Kassignvariableop_62_adam_tic_tac_toe_neural_net_batch_normalization_gamma_m:	Y
Jassignvariableop_63_adam_tic_tac_toe_neural_net_batch_normalization_beta_m:	]
Aassignvariableop_64_adam_tic_tac_toe_neural_net_conv2d_1_kernel_m:N
?assignvariableop_65_adam_tic_tac_toe_neural_net_conv2d_1_bias_m:	\
Massignvariableop_66_adam_tic_tac_toe_neural_net_batch_normalization_1_gamma_m:	[
Lassignvariableop_67_adam_tic_tac_toe_neural_net_batch_normalization_1_beta_m:	]
Aassignvariableop_68_adam_tic_tac_toe_neural_net_conv2d_2_kernel_m:N
?assignvariableop_69_adam_tic_tac_toe_neural_net_conv2d_2_bias_m:	\
Massignvariableop_70_adam_tic_tac_toe_neural_net_batch_normalization_2_gamma_m:	[
Lassignvariableop_71_adam_tic_tac_toe_neural_net_batch_normalization_2_beta_m:	]
Aassignvariableop_72_adam_tic_tac_toe_neural_net_conv2d_3_kernel_m:N
?assignvariableop_73_adam_tic_tac_toe_neural_net_conv2d_3_bias_m:	\
Massignvariableop_74_adam_tic_tac_toe_neural_net_batch_normalization_3_gamma_m:	[
Lassignvariableop_75_adam_tic_tac_toe_neural_net_batch_normalization_3_beta_m:	R
>assignvariableop_76_adam_tic_tac_toe_neural_net_dense_kernel_m:
K
<assignvariableop_77_adam_tic_tac_toe_neural_net_dense_bias_m:	\
Massignvariableop_78_adam_tic_tac_toe_neural_net_batch_normalization_4_gamma_m:	[
Lassignvariableop_79_adam_tic_tac_toe_neural_net_batch_normalization_4_beta_m:	T
@assignvariableop_80_adam_tic_tac_toe_neural_net_dense_1_kernel_m:
M
>assignvariableop_81_adam_tic_tac_toe_neural_net_dense_1_bias_m:	\
Massignvariableop_82_adam_tic_tac_toe_neural_net_batch_normalization_5_gamma_m:	[
Lassignvariableop_83_adam_tic_tac_toe_neural_net_batch_normalization_5_beta_m:	S
@assignvariableop_84_adam_tic_tac_toe_neural_net_dense_2_kernel_m:	2L
>assignvariableop_85_adam_tic_tac_toe_neural_net_dense_2_bias_m:2R
@assignvariableop_86_adam_tic_tac_toe_neural_net_dense_3_kernel_m:2L
>assignvariableop_87_adam_tic_tac_toe_neural_net_dense_3_bias_m:R
@assignvariableop_88_adam_tic_tac_toe_neural_net_dense_4_kernel_m:L
>assignvariableop_89_adam_tic_tac_toe_neural_net_dense_4_bias_m:R
@assignvariableop_90_adam_tic_tac_toe_neural_net_dense_5_kernel_m:	L
>assignvariableop_91_adam_tic_tac_toe_neural_net_dense_5_bias_m:	S
@assignvariableop_92_adam_tic_tac_toe_neural_net_dense_6_kernel_m:	2L
>assignvariableop_93_adam_tic_tac_toe_neural_net_dense_6_bias_m:2R
@assignvariableop_94_adam_tic_tac_toe_neural_net_dense_7_kernel_m:2(L
>assignvariableop_95_adam_tic_tac_toe_neural_net_dense_7_bias_m:(R
@assignvariableop_96_adam_tic_tac_toe_neural_net_dense_8_kernel_m:(L
>assignvariableop_97_adam_tic_tac_toe_neural_net_dense_8_bias_m:Z
?assignvariableop_98_adam_tic_tac_toe_neural_net_conv2d_kernel_v:L
=assignvariableop_99_adam_tic_tac_toe_neural_net_conv2d_bias_v:	[
Lassignvariableop_100_adam_tic_tac_toe_neural_net_batch_normalization_gamma_v:	Z
Kassignvariableop_101_adam_tic_tac_toe_neural_net_batch_normalization_beta_v:	^
Bassignvariableop_102_adam_tic_tac_toe_neural_net_conv2d_1_kernel_v:O
@assignvariableop_103_adam_tic_tac_toe_neural_net_conv2d_1_bias_v:	]
Nassignvariableop_104_adam_tic_tac_toe_neural_net_batch_normalization_1_gamma_v:	\
Massignvariableop_105_adam_tic_tac_toe_neural_net_batch_normalization_1_beta_v:	^
Bassignvariableop_106_adam_tic_tac_toe_neural_net_conv2d_2_kernel_v:O
@assignvariableop_107_adam_tic_tac_toe_neural_net_conv2d_2_bias_v:	]
Nassignvariableop_108_adam_tic_tac_toe_neural_net_batch_normalization_2_gamma_v:	\
Massignvariableop_109_adam_tic_tac_toe_neural_net_batch_normalization_2_beta_v:	^
Bassignvariableop_110_adam_tic_tac_toe_neural_net_conv2d_3_kernel_v:O
@assignvariableop_111_adam_tic_tac_toe_neural_net_conv2d_3_bias_v:	]
Nassignvariableop_112_adam_tic_tac_toe_neural_net_batch_normalization_3_gamma_v:	\
Massignvariableop_113_adam_tic_tac_toe_neural_net_batch_normalization_3_beta_v:	S
?assignvariableop_114_adam_tic_tac_toe_neural_net_dense_kernel_v:
L
=assignvariableop_115_adam_tic_tac_toe_neural_net_dense_bias_v:	]
Nassignvariableop_116_adam_tic_tac_toe_neural_net_batch_normalization_4_gamma_v:	\
Massignvariableop_117_adam_tic_tac_toe_neural_net_batch_normalization_4_beta_v:	U
Aassignvariableop_118_adam_tic_tac_toe_neural_net_dense_1_kernel_v:
N
?assignvariableop_119_adam_tic_tac_toe_neural_net_dense_1_bias_v:	]
Nassignvariableop_120_adam_tic_tac_toe_neural_net_batch_normalization_5_gamma_v:	\
Massignvariableop_121_adam_tic_tac_toe_neural_net_batch_normalization_5_beta_v:	T
Aassignvariableop_122_adam_tic_tac_toe_neural_net_dense_2_kernel_v:	2M
?assignvariableop_123_adam_tic_tac_toe_neural_net_dense_2_bias_v:2S
Aassignvariableop_124_adam_tic_tac_toe_neural_net_dense_3_kernel_v:2M
?assignvariableop_125_adam_tic_tac_toe_neural_net_dense_3_bias_v:S
Aassignvariableop_126_adam_tic_tac_toe_neural_net_dense_4_kernel_v:M
?assignvariableop_127_adam_tic_tac_toe_neural_net_dense_4_bias_v:S
Aassignvariableop_128_adam_tic_tac_toe_neural_net_dense_5_kernel_v:	M
?assignvariableop_129_adam_tic_tac_toe_neural_net_dense_5_bias_v:	T
Aassignvariableop_130_adam_tic_tac_toe_neural_net_dense_6_kernel_v:	2M
?assignvariableop_131_adam_tic_tac_toe_neural_net_dense_6_bias_v:2S
Aassignvariableop_132_adam_tic_tac_toe_neural_net_dense_7_kernel_v:2(M
?assignvariableop_133_adam_tic_tac_toe_neural_net_dense_7_bias_v:(S
Aassignvariableop_134_adam_tic_tac_toe_neural_net_dense_8_kernel_v:(M
?assignvariableop_135_adam_tic_tac_toe_neural_net_dense_8_bias_v:
identity_137¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_100¢AssignVariableOp_101¢AssignVariableOp_102¢AssignVariableOp_103¢AssignVariableOp_104¢AssignVariableOp_105¢AssignVariableOp_106¢AssignVariableOp_107¢AssignVariableOp_108¢AssignVariableOp_109¢AssignVariableOp_11¢AssignVariableOp_110¢AssignVariableOp_111¢AssignVariableOp_112¢AssignVariableOp_113¢AssignVariableOp_114¢AssignVariableOp_115¢AssignVariableOp_116¢AssignVariableOp_117¢AssignVariableOp_118¢AssignVariableOp_119¢AssignVariableOp_12¢AssignVariableOp_120¢AssignVariableOp_121¢AssignVariableOp_122¢AssignVariableOp_123¢AssignVariableOp_124¢AssignVariableOp_125¢AssignVariableOp_126¢AssignVariableOp_127¢AssignVariableOp_128¢AssignVariableOp_129¢AssignVariableOp_13¢AssignVariableOp_130¢AssignVariableOp_131¢AssignVariableOp_132¢AssignVariableOp_133¢AssignVariableOp_134¢AssignVariableOp_135¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_90¢AssignVariableOp_91¢AssignVariableOp_92¢AssignVariableOp_93¢AssignVariableOp_94¢AssignVariableOp_95¢AssignVariableOp_96¢AssignVariableOp_97¢AssignVariableOp_98¢AssignVariableOp_99=
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*¿<
valueµ<B²<B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/44/.ATTRIBUTES/VARIABLE_VALUEB'variables/45/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB'variables/50/.ATTRIBUTES/VARIABLE_VALUEB'variables/51/.ATTRIBUTES/VARIABLE_VALUEB'variables/52/.ATTRIBUTES/VARIABLE_VALUEB'variables/53/.ATTRIBUTES/VARIABLE_VALUEB'variables/54/.ATTRIBUTES/VARIABLE_VALUEB'variables/55/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/40/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/41/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/42/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/43/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/44/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/45/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/46/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/47/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/48/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/49/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/38/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/39/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/40/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/41/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/42/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/43/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/44/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/45/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/46/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/47/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/48/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/49/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*¨
valueBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ò
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*º
_output_shapes§
¤:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOpAssignVariableOp5assignvariableop_tic_tac_toe_neural_net_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_1AssignVariableOp5assignvariableop_1_tic_tac_toe_neural_net_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_2AssignVariableOpCassignvariableop_2_tic_tac_toe_neural_net_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_3AssignVariableOpBassignvariableop_3_tic_tac_toe_neural_net_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_4AssignVariableOpIassignvariableop_4_tic_tac_toe_neural_net_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_5AssignVariableOpMassignvariableop_5_tic_tac_toe_neural_net_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_6AssignVariableOp9assignvariableop_6_tic_tac_toe_neural_net_conv2d_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_7AssignVariableOp7assignvariableop_7_tic_tac_toe_neural_net_conv2d_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_8AssignVariableOpEassignvariableop_8_tic_tac_toe_neural_net_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_9AssignVariableOpDassignvariableop_9_tic_tac_toe_neural_net_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_10AssignVariableOpLassignvariableop_10_tic_tac_toe_neural_net_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_11AssignVariableOpPassignvariableop_11_tic_tac_toe_neural_net_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_12AssignVariableOp:assignvariableop_12_tic_tac_toe_neural_net_conv2d_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_13AssignVariableOp8assignvariableop_13_tic_tac_toe_neural_net_conv2d_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_14AssignVariableOpFassignvariableop_14_tic_tac_toe_neural_net_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_15AssignVariableOpEassignvariableop_15_tic_tac_toe_neural_net_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_16AssignVariableOpLassignvariableop_16_tic_tac_toe_neural_net_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_17AssignVariableOpPassignvariableop_17_tic_tac_toe_neural_net_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_18AssignVariableOp:assignvariableop_18_tic_tac_toe_neural_net_conv2d_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_19AssignVariableOp8assignvariableop_19_tic_tac_toe_neural_net_conv2d_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_20AssignVariableOpFassignvariableop_20_tic_tac_toe_neural_net_batch_normalization_3_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_21AssignVariableOpEassignvariableop_21_tic_tac_toe_neural_net_batch_normalization_3_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_22AssignVariableOpLassignvariableop_22_tic_tac_toe_neural_net_batch_normalization_3_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_23AssignVariableOpPassignvariableop_23_tic_tac_toe_neural_net_batch_normalization_3_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_24AssignVariableOp7assignvariableop_24_tic_tac_toe_neural_net_dense_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_25AssignVariableOp5assignvariableop_25_tic_tac_toe_neural_net_dense_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_26AssignVariableOpFassignvariableop_26_tic_tac_toe_neural_net_batch_normalization_4_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_27AssignVariableOpEassignvariableop_27_tic_tac_toe_neural_net_batch_normalization_4_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_28AssignVariableOpLassignvariableop_28_tic_tac_toe_neural_net_batch_normalization_4_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_29AssignVariableOpPassignvariableop_29_tic_tac_toe_neural_net_batch_normalization_4_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_30AssignVariableOp9assignvariableop_30_tic_tac_toe_neural_net_dense_1_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_31AssignVariableOp7assignvariableop_31_tic_tac_toe_neural_net_dense_1_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_32AssignVariableOpFassignvariableop_32_tic_tac_toe_neural_net_batch_normalization_5_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_33AssignVariableOpEassignvariableop_33_tic_tac_toe_neural_net_batch_normalization_5_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_34AssignVariableOpLassignvariableop_34_tic_tac_toe_neural_net_batch_normalization_5_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_35AssignVariableOpPassignvariableop_35_tic_tac_toe_neural_net_batch_normalization_5_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_36AssignVariableOp9assignvariableop_36_tic_tac_toe_neural_net_dense_2_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_37AssignVariableOp7assignvariableop_37_tic_tac_toe_neural_net_dense_2_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_38AssignVariableOp9assignvariableop_38_tic_tac_toe_neural_net_dense_3_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_39AssignVariableOp7assignvariableop_39_tic_tac_toe_neural_net_dense_3_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_40AssignVariableOp9assignvariableop_40_tic_tac_toe_neural_net_dense_4_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_41AssignVariableOp7assignvariableop_41_tic_tac_toe_neural_net_dense_4_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_42AssignVariableOp9assignvariableop_42_tic_tac_toe_neural_net_dense_5_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_43AssignVariableOp7assignvariableop_43_tic_tac_toe_neural_net_dense_5_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_44AssignVariableOp9assignvariableop_44_tic_tac_toe_neural_net_dense_6_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_45AssignVariableOp7assignvariableop_45_tic_tac_toe_neural_net_dense_6_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_46AssignVariableOp9assignvariableop_46_tic_tac_toe_neural_net_dense_7_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_47AssignVariableOp7assignvariableop_47_tic_tac_toe_neural_net_dense_7_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_48AssignVariableOp9assignvariableop_48_tic_tac_toe_neural_net_dense_8_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_49AssignVariableOp7assignvariableop_49_tic_tac_toe_neural_net_dense_8_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOpassignvariableop_50_total_2Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOpassignvariableop_51_count_2Identity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOpassignvariableop_52_total_1Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_53AssignVariableOpassignvariableop_53_count_1Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_54AssignVariableOpassignvariableop_54_totalIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_55AssignVariableOpassignvariableop_55_countIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_56AssignVariableOpassignvariableop_56_adam_iterIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOpassignvariableop_57_adam_beta_1Identity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOpassignvariableop_58_adam_beta_2Identity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_59AssignVariableOpassignvariableop_59_adam_decayIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_60AssignVariableOp?assignvariableop_60_adam_tic_tac_toe_neural_net_conv2d_kernel_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_61AssignVariableOp=assignvariableop_61_adam_tic_tac_toe_neural_net_conv2d_bias_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_62AssignVariableOpKassignvariableop_62_adam_tic_tac_toe_neural_net_batch_normalization_gamma_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_63AssignVariableOpJassignvariableop_63_adam_tic_tac_toe_neural_net_batch_normalization_beta_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_64AssignVariableOpAassignvariableop_64_adam_tic_tac_toe_neural_net_conv2d_1_kernel_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_65AssignVariableOp?assignvariableop_65_adam_tic_tac_toe_neural_net_conv2d_1_bias_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_66AssignVariableOpMassignvariableop_66_adam_tic_tac_toe_neural_net_batch_normalization_1_gamma_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_67AssignVariableOpLassignvariableop_67_adam_tic_tac_toe_neural_net_batch_normalization_1_beta_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_68AssignVariableOpAassignvariableop_68_adam_tic_tac_toe_neural_net_conv2d_2_kernel_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_69AssignVariableOp?assignvariableop_69_adam_tic_tac_toe_neural_net_conv2d_2_bias_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_70AssignVariableOpMassignvariableop_70_adam_tic_tac_toe_neural_net_batch_normalization_2_gamma_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_71AssignVariableOpLassignvariableop_71_adam_tic_tac_toe_neural_net_batch_normalization_2_beta_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_72AssignVariableOpAassignvariableop_72_adam_tic_tac_toe_neural_net_conv2d_3_kernel_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_73AssignVariableOp?assignvariableop_73_adam_tic_tac_toe_neural_net_conv2d_3_bias_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_74AssignVariableOpMassignvariableop_74_adam_tic_tac_toe_neural_net_batch_normalization_3_gamma_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_75AssignVariableOpLassignvariableop_75_adam_tic_tac_toe_neural_net_batch_normalization_3_beta_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_76AssignVariableOp>assignvariableop_76_adam_tic_tac_toe_neural_net_dense_kernel_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_77AssignVariableOp<assignvariableop_77_adam_tic_tac_toe_neural_net_dense_bias_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_78AssignVariableOpMassignvariableop_78_adam_tic_tac_toe_neural_net_batch_normalization_4_gamma_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_79AssignVariableOpLassignvariableop_79_adam_tic_tac_toe_neural_net_batch_normalization_4_beta_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_80AssignVariableOp@assignvariableop_80_adam_tic_tac_toe_neural_net_dense_1_kernel_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_81AssignVariableOp>assignvariableop_81_adam_tic_tac_toe_neural_net_dense_1_bias_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_82AssignVariableOpMassignvariableop_82_adam_tic_tac_toe_neural_net_batch_normalization_5_gamma_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_83AssignVariableOpLassignvariableop_83_adam_tic_tac_toe_neural_net_batch_normalization_5_beta_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_84AssignVariableOp@assignvariableop_84_adam_tic_tac_toe_neural_net_dense_2_kernel_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_85AssignVariableOp>assignvariableop_85_adam_tic_tac_toe_neural_net_dense_2_bias_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_86AssignVariableOp@assignvariableop_86_adam_tic_tac_toe_neural_net_dense_3_kernel_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_87AssignVariableOp>assignvariableop_87_adam_tic_tac_toe_neural_net_dense_3_bias_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_88AssignVariableOp@assignvariableop_88_adam_tic_tac_toe_neural_net_dense_4_kernel_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_89AssignVariableOp>assignvariableop_89_adam_tic_tac_toe_neural_net_dense_4_bias_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_90AssignVariableOp@assignvariableop_90_adam_tic_tac_toe_neural_net_dense_5_kernel_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_91AssignVariableOp>assignvariableop_91_adam_tic_tac_toe_neural_net_dense_5_bias_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_92AssignVariableOp@assignvariableop_92_adam_tic_tac_toe_neural_net_dense_6_kernel_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_93AssignVariableOp>assignvariableop_93_adam_tic_tac_toe_neural_net_dense_6_bias_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_94AssignVariableOp@assignvariableop_94_adam_tic_tac_toe_neural_net_dense_7_kernel_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_95AssignVariableOp>assignvariableop_95_adam_tic_tac_toe_neural_net_dense_7_bias_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_96AssignVariableOp@assignvariableop_96_adam_tic_tac_toe_neural_net_dense_8_kernel_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_97AssignVariableOp>assignvariableop_97_adam_tic_tac_toe_neural_net_dense_8_bias_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_98AssignVariableOp?assignvariableop_98_adam_tic_tac_toe_neural_net_conv2d_kernel_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_99AssignVariableOp=assignvariableop_99_adam_tic_tac_toe_neural_net_conv2d_bias_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_100AssignVariableOpLassignvariableop_100_adam_tic_tac_toe_neural_net_batch_normalization_gamma_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_101AssignVariableOpKassignvariableop_101_adam_tic_tac_toe_neural_net_batch_normalization_beta_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_102AssignVariableOpBassignvariableop_102_adam_tic_tac_toe_neural_net_conv2d_1_kernel_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_103AssignVariableOp@assignvariableop_103_adam_tic_tac_toe_neural_net_conv2d_1_bias_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_104AssignVariableOpNassignvariableop_104_adam_tic_tac_toe_neural_net_batch_normalization_1_gamma_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_105AssignVariableOpMassignvariableop_105_adam_tic_tac_toe_neural_net_batch_normalization_1_beta_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_106AssignVariableOpBassignvariableop_106_adam_tic_tac_toe_neural_net_conv2d_2_kernel_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_107AssignVariableOp@assignvariableop_107_adam_tic_tac_toe_neural_net_conv2d_2_bias_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_108AssignVariableOpNassignvariableop_108_adam_tic_tac_toe_neural_net_batch_normalization_2_gamma_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_109AssignVariableOpMassignvariableop_109_adam_tic_tac_toe_neural_net_batch_normalization_2_beta_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_110AssignVariableOpBassignvariableop_110_adam_tic_tac_toe_neural_net_conv2d_3_kernel_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_111AssignVariableOp@assignvariableop_111_adam_tic_tac_toe_neural_net_conv2d_3_bias_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_112AssignVariableOpNassignvariableop_112_adam_tic_tac_toe_neural_net_batch_normalization_3_gamma_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_113AssignVariableOpMassignvariableop_113_adam_tic_tac_toe_neural_net_batch_normalization_3_beta_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_114AssignVariableOp?assignvariableop_114_adam_tic_tac_toe_neural_net_dense_kernel_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_115AssignVariableOp=assignvariableop_115_adam_tic_tac_toe_neural_net_dense_bias_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_116AssignVariableOpNassignvariableop_116_adam_tic_tac_toe_neural_net_batch_normalization_4_gamma_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_117AssignVariableOpMassignvariableop_117_adam_tic_tac_toe_neural_net_batch_normalization_4_beta_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_118AssignVariableOpAassignvariableop_118_adam_tic_tac_toe_neural_net_dense_1_kernel_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_119AssignVariableOp?assignvariableop_119_adam_tic_tac_toe_neural_net_dense_1_bias_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_120AssignVariableOpNassignvariableop_120_adam_tic_tac_toe_neural_net_batch_normalization_5_gamma_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_121AssignVariableOpMassignvariableop_121_adam_tic_tac_toe_neural_net_batch_normalization_5_beta_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_122AssignVariableOpAassignvariableop_122_adam_tic_tac_toe_neural_net_dense_2_kernel_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_123AssignVariableOp?assignvariableop_123_adam_tic_tac_toe_neural_net_dense_2_bias_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_124AssignVariableOpAassignvariableop_124_adam_tic_tac_toe_neural_net_dense_3_kernel_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_125AssignVariableOp?assignvariableop_125_adam_tic_tac_toe_neural_net_dense_3_bias_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_126AssignVariableOpAassignvariableop_126_adam_tic_tac_toe_neural_net_dense_4_kernel_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_127AssignVariableOp?assignvariableop_127_adam_tic_tac_toe_neural_net_dense_4_bias_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_128AssignVariableOpAassignvariableop_128_adam_tic_tac_toe_neural_net_dense_5_kernel_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_129AssignVariableOp?assignvariableop_129_adam_tic_tac_toe_neural_net_dense_5_bias_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_130AssignVariableOpAassignvariableop_130_adam_tic_tac_toe_neural_net_dense_6_kernel_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_131AssignVariableOp?assignvariableop_131_adam_tic_tac_toe_neural_net_dense_6_bias_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_132AssignVariableOpAassignvariableop_132_adam_tic_tac_toe_neural_net_dense_7_kernel_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_133AssignVariableOp?assignvariableop_133_adam_tic_tac_toe_neural_net_dense_7_bias_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_134AssignVariableOpAassignvariableop_134_adam_tic_tac_toe_neural_net_dense_8_kernel_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_135AssignVariableOp?assignvariableop_135_adam_tic_tac_toe_neural_net_dense_8_bias_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ¤
Identity_136Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_137IdentityIdentity_136:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_137Identity_137:output:0*§
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
£
F
*__inference_dropout_layer_call_fn_41075040

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_41072377a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó	
ù
E__inference_dense_1_layer_call_and_return_conditional_losses_41074905

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

÷
E__inference_dense_6_layer_call_and_return_conditional_losses_41075247

inputs1
matmul_readvariableop_resource:	2-
biasadd_readvariableop_resource:2
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
c
E__inference_dropout_layer_call_and_return_conditional_losses_41072455

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´


F__inference_conv2d_1_layer_call_and_return_conditional_losses_41074552

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡

ö
E__inference_dense_5_layer_call_and_return_conditional_losses_41072491

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
c
E__inference_dropout_layer_call_and_return_conditional_losses_41072478

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¿L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ñ
serving_defaultÝ
C
input_18
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ	<
output_20
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:á

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
L1_conv
	L1_conv_batch_norm

L2_conv
L2_conv_batch_norm
L3_conv
L3_conv_batch_norm
L4_conv
L4_conv_batch_norm

L1_flatten
L1_dense
L1_dense_batch_norm
L2_dense
L2_dense_batch_norm
dropout
L1_policy_dense
L2_policy_dense
L3_policy_dense

policy_out
L1_value
L2_value
	value_out
	optimizer
policy_loss_metric
value_loss_metric
 loss_metric
!
signatures"
_tf_keras_model
Ö
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
216
317
418
519
620
721
822
923
:24
;25
<26
=27
>28
?29
@30
A31
B32
C33
D34
E35
F36
G37
H38
I39
J40
K41
L42
M43
N44
O45
P46
Q47
R48
S49
T50
U51
V52
W53
X54
Y55"
trackable_list_wrapper
Æ
"0
#1
$2
%3
(4
)5
*6
+7
.8
/9
010
111
412
513
614
715
:16
;17
<18
=19
@20
A21
B22
C23
F24
G25
H26
I27
J28
K29
L30
M31
N32
O33
P34
Q35
R36
S37"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object

_trace_0
`trace_1
atrace_2
btrace_32
9__inference_tic_tac_toe_neural_net_layer_call_fn_41072662
9__inference_tic_tac_toe_neural_net_layer_call_fn_41073882
9__inference_tic_tac_toe_neural_net_layer_call_fn_41073989
9__inference_tic_tac_toe_neural_net_layer_call_fn_41073390¯
¦²¢
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z_trace_0z`trace_1zatrace_2zbtrace_3
õ
ctrace_0
dtrace_1
etrace_2
ftrace_32
T__inference_tic_tac_toe_neural_net_layer_call_and_return_conditional_losses_41074182
T__inference_tic_tac_toe_neural_net_layer_call_and_return_conditional_losses_41074452
T__inference_tic_tac_toe_neural_net_layer_call_and_return_conditional_losses_41073528
T__inference_tic_tac_toe_neural_net_layer_call_and_return_conditional_losses_41073666¯
¦²¢
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zctrace_0zdtrace_1zetrace_2zftrace_3
ÎBË
#__inference__wrapped_model_41071806input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ý
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses

"kernel
#bias
 m_jit_compiled_convolution_op"
_tf_keras_layer
ê
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses
taxis
	$gamma
%beta
&moving_mean
'moving_variance"
_tf_keras_layer
Ý
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses

(kernel
)bias
 {_jit_compiled_convolution_op"
_tf_keras_layer
í
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis
	*gamma
+beta
,moving_mean
-moving_variance"
_tf_keras_layer
ä
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

.kernel
/bias
!_jit_compiled_convolution_op"
_tf_keras_layer
ñ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis
	0gamma
1beta
2moving_mean
3moving_variance"
_tf_keras_layer
ä
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

4kernel
5bias
!_jit_compiled_convolution_op"
_tf_keras_layer
ñ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis
	6gamma
7beta
8moving_mean
9moving_variance"
_tf_keras_layer
«
	variables
 trainable_variables
¡regularization_losses
¢	keras_api
£__call__
+¤&call_and_return_all_conditional_losses"
_tf_keras_layer
Á
¥	variables
¦trainable_variables
§regularization_losses
¨	keras_api
©__call__
+ª&call_and_return_all_conditional_losses

:kernel
;bias"
_tf_keras_layer
ñ
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
¯__call__
+°&call_and_return_all_conditional_losses
	±axis
	<gamma
=beta
>moving_mean
?moving_variance"
_tf_keras_layer
Á
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
¶__call__
+·&call_and_return_all_conditional_losses

@kernel
Abias"
_tf_keras_layer
ñ
¸	variables
¹trainable_variables
ºregularization_losses
»	keras_api
¼__call__
+½&call_and_return_all_conditional_losses
	¾axis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance"
_tf_keras_layer
Ã
¿	variables
Àtrainable_variables
Áregularization_losses
Â	keras_api
Ã__call__
+Ä&call_and_return_all_conditional_losses
Å_random_generator"
_tf_keras_layer
Á
Æ	variables
Çtrainable_variables
Èregularization_losses
É	keras_api
Ê__call__
+Ë&call_and_return_all_conditional_losses

Fkernel
Gbias"
_tf_keras_layer
Á
Ì	variables
Ítrainable_variables
Îregularization_losses
Ï	keras_api
Ð__call__
+Ñ&call_and_return_all_conditional_losses

Hkernel
Ibias"
_tf_keras_layer
Á
Ò	variables
Ótrainable_variables
Ôregularization_losses
Õ	keras_api
Ö__call__
+×&call_and_return_all_conditional_losses

Jkernel
Kbias"
_tf_keras_layer
Á
Ø	variables
Ùtrainable_variables
Úregularization_losses
Û	keras_api
Ü__call__
+Ý&call_and_return_all_conditional_losses

Lkernel
Mbias"
_tf_keras_layer
Á
Þ	variables
ßtrainable_variables
àregularization_losses
á	keras_api
â__call__
+ã&call_and_return_all_conditional_losses

Nkernel
Obias"
_tf_keras_layer
Á
ä	variables
åtrainable_variables
æregularization_losses
ç	keras_api
è__call__
+é&call_and_return_all_conditional_losses

Pkernel
Qbias"
_tf_keras_layer
Á
ê	variables
ëtrainable_variables
ìregularization_losses
í	keras_api
î__call__
+ï&call_and_return_all_conditional_losses

Rkernel
Sbias"
_tf_keras_layer
¼
	ðiter
ñbeta_1
òbeta_2

ódecay"m°#m±$m²%m³(m´)mµ*m¶+m·.m¸/m¹0mº1m»4m¼5m½6m¾7m¿:mÀ;mÁ<mÂ=mÃ@mÄAmÅBmÆCmÇFmÈGmÉHmÊImËJmÌKmÍLmÎMmÏNmÐOmÑPmÒQmÓRmÔSmÕ"vÖ#v×$vØ%vÙ(vÚ)vÛ*vÜ+vÝ.vÞ/vß0và1vá4vâ5vã6vä7vå:væ;vç<vè=vé@vêAvëBvìCvíFvîGvïHvðIvñJvòKvóLvôMvõNvöOv÷PvøQvùRvúSvû"
	optimizer
P
ô	variables
õ	keras_api
	Ttotal
	Ucount"
_tf_keras_metric
P
ö	variables
÷	keras_api
	Vtotal
	Wcount"
_tf_keras_metric
P
ø	variables
ù	keras_api
	Xtotal
	Ycount"
_tf_keras_metric
-
úserving_default"
signature_map
?:=2$tic_tac_toe_neural_net/conv2d/kernel
1:/2"tic_tac_toe_neural_net/conv2d/bias
?:=20tic_tac_toe_neural_net/batch_normalization/gamma
>:<2/tic_tac_toe_neural_net/batch_normalization/beta
G:E (26tic_tac_toe_neural_net/batch_normalization/moving_mean
K:I (2:tic_tac_toe_neural_net/batch_normalization/moving_variance
B:@2&tic_tac_toe_neural_net/conv2d_1/kernel
3:12$tic_tac_toe_neural_net/conv2d_1/bias
A:?22tic_tac_toe_neural_net/batch_normalization_1/gamma
@:>21tic_tac_toe_neural_net/batch_normalization_1/beta
I:G (28tic_tac_toe_neural_net/batch_normalization_1/moving_mean
M:K (2<tic_tac_toe_neural_net/batch_normalization_1/moving_variance
B:@2&tic_tac_toe_neural_net/conv2d_2/kernel
3:12$tic_tac_toe_neural_net/conv2d_2/bias
A:?22tic_tac_toe_neural_net/batch_normalization_2/gamma
@:>21tic_tac_toe_neural_net/batch_normalization_2/beta
I:G (28tic_tac_toe_neural_net/batch_normalization_2/moving_mean
M:K (2<tic_tac_toe_neural_net/batch_normalization_2/moving_variance
B:@2&tic_tac_toe_neural_net/conv2d_3/kernel
3:12$tic_tac_toe_neural_net/conv2d_3/bias
A:?22tic_tac_toe_neural_net/batch_normalization_3/gamma
@:>21tic_tac_toe_neural_net/batch_normalization_3/beta
I:G (28tic_tac_toe_neural_net/batch_normalization_3/moving_mean
M:K (2<tic_tac_toe_neural_net/batch_normalization_3/moving_variance
7:5
2#tic_tac_toe_neural_net/dense/kernel
0:.2!tic_tac_toe_neural_net/dense/bias
A:?22tic_tac_toe_neural_net/batch_normalization_4/gamma
@:>21tic_tac_toe_neural_net/batch_normalization_4/beta
I:G (28tic_tac_toe_neural_net/batch_normalization_4/moving_mean
M:K (2<tic_tac_toe_neural_net/batch_normalization_4/moving_variance
9:7
2%tic_tac_toe_neural_net/dense_1/kernel
2:02#tic_tac_toe_neural_net/dense_1/bias
A:?22tic_tac_toe_neural_net/batch_normalization_5/gamma
@:>21tic_tac_toe_neural_net/batch_normalization_5/beta
I:G (28tic_tac_toe_neural_net/batch_normalization_5/moving_mean
M:K (2<tic_tac_toe_neural_net/batch_normalization_5/moving_variance
8:6	22%tic_tac_toe_neural_net/dense_2/kernel
1:/22#tic_tac_toe_neural_net/dense_2/bias
7:522%tic_tac_toe_neural_net/dense_3/kernel
1:/2#tic_tac_toe_neural_net/dense_3/bias
7:52%tic_tac_toe_neural_net/dense_4/kernel
1:/2#tic_tac_toe_neural_net/dense_4/bias
7:5	2%tic_tac_toe_neural_net/dense_5/kernel
1:/	2#tic_tac_toe_neural_net/dense_5/bias
8:6	22%tic_tac_toe_neural_net/dense_6/kernel
1:/22#tic_tac_toe_neural_net/dense_6/bias
7:52(2%tic_tac_toe_neural_net/dense_7/kernel
1:/(2#tic_tac_toe_neural_net/dense_7/bias
7:5(2%tic_tac_toe_neural_net/dense_8/kernel
1:/2#tic_tac_toe_neural_net/dense_8/bias
:  (2total
:  (2count
:  (2total
:  (2count
:  (2total
:  (2count
¦
&0
'1
,2
-3
24
35
86
97
>8
?9
D10
E11
T12
U13
V14
W15
X16
Y17"
trackable_list_wrapper
¾
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20"
trackable_list_wrapper
5
0
1
 2"
trackable_list_wrapper
 "
trackable_list_wrapper
]
total_policy_loss
total_value_loss
 
total_loss"
trackable_dict_wrapper
ûBø
9__inference_tic_tac_toe_neural_net_layer_call_fn_41072662input_1"¯
¦²¢
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
õBò
9__inference_tic_tac_toe_neural_net_layer_call_fn_41073882x"¯
¦²¢
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
õBò
9__inference_tic_tac_toe_neural_net_layer_call_fn_41073989x"¯
¦²¢
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ûBø
9__inference_tic_tac_toe_neural_net_layer_call_fn_41073390input_1"¯
¦²¢
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
T__inference_tic_tac_toe_neural_net_layer_call_and_return_conditional_losses_41074182x"¯
¦²¢
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
T__inference_tic_tac_toe_neural_net_layer_call_and_return_conditional_losses_41074452x"¯
¦²¢
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
T__inference_tic_tac_toe_neural_net_layer_call_and_return_conditional_losses_41073528input_1"¯
¦²¢
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
T__inference_tic_tac_toe_neural_net_layer_call_and_return_conditional_losses_41073666input_1"¯
¦²¢
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ûnon_trainable_variables
ülayers
ýmetrics
 þlayer_regularization_losses
ÿlayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
ï
trace_02Ð
)__inference_conv2d_layer_call_fn_41074461¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ë
D__inference_conv2d_layer_call_and_return_conditional_losses_41074471¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
<
$0
%1
&2
'3"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
â
trace_0
trace_12§
6__inference_batch_normalization_layer_call_fn_41074484
6__inference_batch_normalization_layer_call_fn_41074497´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12Ý
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_41074515
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_41074533´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
ñ
trace_02Ò
+__inference_conv2d_1_layer_call_fn_41074542¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02í
F__inference_conv2d_1_layer_call_and_return_conditional_losses_41074552¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
<
*0
+1
,2
-3"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
æ
trace_0
trace_12«
8__inference_batch_normalization_1_layer_call_fn_41074565
8__inference_batch_normalization_1_layer_call_fn_41074578´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12á
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_41074596
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_41074614´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ñ
 trace_02Ò
+__inference_conv2d_2_layer_call_fn_41074623¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z trace_0

¡trace_02í
F__inference_conv2d_2_layer_call_and_return_conditional_losses_41074633¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¡trace_0
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
<
00
11
22
33"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
æ
§trace_0
¨trace_12«
8__inference_batch_normalization_2_layer_call_fn_41074646
8__inference_batch_normalization_2_layer_call_fn_41074659´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z§trace_0z¨trace_1

©trace_0
ªtrace_12á
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_41074677
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_41074695´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z©trace_0zªtrace_1
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
¯layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ñ
°trace_02Ò
+__inference_conv2d_3_layer_call_fn_41074704¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z°trace_0

±trace_02í
F__inference_conv2d_3_layer_call_and_return_conditional_losses_41074714¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z±trace_0
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
<
60
71
82
93"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
æ
·trace_0
¸trace_12«
8__inference_batch_normalization_3_layer_call_fn_41074727
8__inference_batch_normalization_3_layer_call_fn_41074740´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z·trace_0z¸trace_1

¹trace_0
ºtrace_12á
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_41074758
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_41074776´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z¹trace_0zºtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
»non_trainable_variables
¼layers
½metrics
 ¾layer_regularization_losses
¿layer_metrics
	variables
 trainable_variables
¡regularization_losses
£__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
ð
Àtrace_02Ñ
*__inference_flatten_layer_call_fn_41074781¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÀtrace_0

Átrace_02ì
E__inference_flatten_layer_call_and_return_conditional_losses_41074787¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÁtrace_0
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
¥	variables
¦trainable_variables
§regularization_losses
©__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses"
_generic_user_object
î
Çtrace_02Ï
(__inference_dense_layer_call_fn_41074796¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÇtrace_0

Ètrace_02ê
C__inference_dense_layer_call_and_return_conditional_losses_41074806¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÈtrace_0
<
<0
=1
>2
?3"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Énon_trainable_variables
Êlayers
Ëmetrics
 Ìlayer_regularization_losses
Ílayer_metrics
«	variables
¬trainable_variables
­regularization_losses
¯__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
æ
Îtrace_0
Ïtrace_12«
8__inference_batch_normalization_4_layer_call_fn_41074819
8__inference_batch_normalization_4_layer_call_fn_41074832´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÎtrace_0zÏtrace_1

Ðtrace_0
Ñtrace_12á
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_41074852
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_41074886´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÐtrace_0zÑtrace_1
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ònon_trainable_variables
Ólayers
Ômetrics
 Õlayer_regularization_losses
Ölayer_metrics
²	variables
³trainable_variables
´regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
ð
×trace_02Ñ
*__inference_dense_1_layer_call_fn_41074895¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z×trace_0

Øtrace_02ì
E__inference_dense_1_layer_call_and_return_conditional_losses_41074905¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zØtrace_0
<
B0
C1
D2
E3"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ùnon_trainable_variables
Úlayers
Ûmetrics
 Ülayer_regularization_losses
Ýlayer_metrics
¸	variables
¹trainable_variables
ºregularization_losses
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
æ
Þtrace_0
ßtrace_12«
8__inference_batch_normalization_5_layer_call_fn_41074918
8__inference_batch_normalization_5_layer_call_fn_41074931´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÞtrace_0zßtrace_1

àtrace_0
átrace_12á
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_41074951
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_41074985´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zàtrace_0zátrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ânon_trainable_variables
ãlayers
ämetrics
 ålayer_regularization_losses
ælayer_metrics
¿	variables
Àtrainable_variables
Áregularization_losses
Ã__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses"
_generic_user_object

çtrace_0
ètrace_1
étrace_2
êtrace_3
ëtrace_4
ìtrace_5
ítrace_6
îtrace_7
ïtrace_8
ðtrace_9
ñtrace_10
òtrace_112Ç
*__inference_dropout_layer_call_fn_41074990
*__inference_dropout_layer_call_fn_41074995
*__inference_dropout_layer_call_fn_41075000
*__inference_dropout_layer_call_fn_41075005
*__inference_dropout_layer_call_fn_41075010
*__inference_dropout_layer_call_fn_41075015
*__inference_dropout_layer_call_fn_41075020
*__inference_dropout_layer_call_fn_41075025
*__inference_dropout_layer_call_fn_41075030
*__inference_dropout_layer_call_fn_41075035
*__inference_dropout_layer_call_fn_41075040
*__inference_dropout_layer_call_fn_41075045´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zçtrace_0zètrace_1zétrace_2zêtrace_3zëtrace_4zìtrace_5zítrace_6zîtrace_7zïtrace_8zðtrace_9zñtrace_10zòtrace_11
â

ótrace_0
ôtrace_1
õtrace_2
ötrace_3
÷trace_4
øtrace_5
ùtrace_6
útrace_7
ûtrace_8
ütrace_9
ýtrace_10
þtrace_112
E__inference_dropout_layer_call_and_return_conditional_losses_41075050
E__inference_dropout_layer_call_and_return_conditional_losses_41075055
E__inference_dropout_layer_call_and_return_conditional_losses_41075060
E__inference_dropout_layer_call_and_return_conditional_losses_41075065
E__inference_dropout_layer_call_and_return_conditional_losses_41075070
E__inference_dropout_layer_call_and_return_conditional_losses_41075075
E__inference_dropout_layer_call_and_return_conditional_losses_41075087
E__inference_dropout_layer_call_and_return_conditional_losses_41075099
E__inference_dropout_layer_call_and_return_conditional_losses_41075111
E__inference_dropout_layer_call_and_return_conditional_losses_41075123
E__inference_dropout_layer_call_and_return_conditional_losses_41075135
E__inference_dropout_layer_call_and_return_conditional_losses_41075147´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zótrace_0zôtrace_1zõtrace_2zötrace_3z÷trace_4zøtrace_5zùtrace_6zútrace_7zûtrace_8zütrace_9zýtrace_10zþtrace_11
"
_generic_user_object
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ÿnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Æ	variables
Çtrainable_variables
Èregularization_losses
Ê__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
ð
trace_02Ñ
*__inference_dense_2_layer_call_fn_41075156¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ì
E__inference_dense_2_layer_call_and_return_conditional_losses_41075167¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ì	variables
Ítrainable_variables
Îregularization_losses
Ð__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses"
_generic_user_object
ð
trace_02Ñ
*__inference_dense_3_layer_call_fn_41075176¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ì
E__inference_dense_3_layer_call_and_return_conditional_losses_41075187¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ò	variables
Ótrainable_variables
Ôregularization_losses
Ö__call__
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses"
_generic_user_object
ð
trace_02Ñ
*__inference_dense_4_layer_call_fn_41075196¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ì
E__inference_dense_4_layer_call_and_return_conditional_losses_41075207¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ø	variables
Ùtrainable_variables
Úregularization_losses
Ü__call__
+Ý&call_and_return_all_conditional_losses
'Ý"call_and_return_conditional_losses"
_generic_user_object
ð
trace_02Ñ
*__inference_dense_5_layer_call_fn_41075216¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ì
E__inference_dense_5_layer_call_and_return_conditional_losses_41075227¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Þ	variables
ßtrainable_variables
àregularization_losses
â__call__
+ã&call_and_return_all_conditional_losses
'ã"call_and_return_conditional_losses"
_generic_user_object
ð
 trace_02Ñ
*__inference_dense_6_layer_call_fn_41075236¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z trace_0

¡trace_02ì
E__inference_dense_6_layer_call_and_return_conditional_losses_41075247¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¡trace_0
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
ä	variables
åtrainable_variables
æregularization_losses
è__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses"
_generic_user_object
ð
§trace_02Ñ
*__inference_dense_7_layer_call_fn_41075256¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z§trace_0

¨trace_02ì
E__inference_dense_7_layer_call_and_return_conditional_losses_41075267¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¨trace_0
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
ê	variables
ëtrainable_variables
ìregularization_losses
î__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses"
_generic_user_object
ð
®trace_02Ñ
*__inference_dense_8_layer_call_fn_41075276¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z®trace_0

¯trace_02ì
E__inference_dense_8_layer_call_and_return_conditional_losses_41075287¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¯trace_0
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
.
T0
U1"
trackable_list_wrapper
.
ô	variables"
_generic_user_object
.
V0
W1"
trackable_list_wrapper
.
ö	variables"
_generic_user_object
.
X0
Y1"
trackable_list_wrapper
.
ø	variables"
_generic_user_object
ÍBÊ
&__inference_signature_wrapper_41073775input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_conv2d_layer_call_fn_41074461inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_conv2d_layer_call_and_return_conditional_losses_41074471inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
üBù
6__inference_batch_normalization_layer_call_fn_41074484inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
üBù
6__inference_batch_normalization_layer_call_fn_41074497inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_41074515inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_41074533inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_conv2d_1_layer_call_fn_41074542inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_conv2d_1_layer_call_and_return_conditional_losses_41074552inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
þBû
8__inference_batch_normalization_1_layer_call_fn_41074565inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
þBû
8__inference_batch_normalization_1_layer_call_fn_41074578inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_41074596inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_41074614inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_conv2d_2_layer_call_fn_41074623inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_conv2d_2_layer_call_and_return_conditional_losses_41074633inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
þBû
8__inference_batch_normalization_2_layer_call_fn_41074646inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
þBû
8__inference_batch_normalization_2_layer_call_fn_41074659inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_41074677inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_41074695inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_conv2d_3_layer_call_fn_41074704inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_conv2d_3_layer_call_and_return_conditional_losses_41074714inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
þBû
8__inference_batch_normalization_3_layer_call_fn_41074727inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
þBû
8__inference_batch_normalization_3_layer_call_fn_41074740inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_41074758inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_41074776inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_flatten_layer_call_fn_41074781inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_flatten_layer_call_and_return_conditional_losses_41074787inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÜBÙ
(__inference_dense_layer_call_fn_41074796inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_dense_layer_call_and_return_conditional_losses_41074806inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
þBû
8__inference_batch_normalization_4_layer_call_fn_41074819inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
þBû
8__inference_batch_normalization_4_layer_call_fn_41074832inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_41074852inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_41074886inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_dense_1_layer_call_fn_41074895inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_dense_1_layer_call_and_return_conditional_losses_41074905inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
þBû
8__inference_batch_normalization_5_layer_call_fn_41074918inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
þBû
8__inference_batch_normalization_5_layer_call_fn_41074931inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_41074951inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_41074985inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ðBí
*__inference_dropout_layer_call_fn_41074990inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ðBí
*__inference_dropout_layer_call_fn_41074995inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ðBí
*__inference_dropout_layer_call_fn_41075000inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ðBí
*__inference_dropout_layer_call_fn_41075005inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ðBí
*__inference_dropout_layer_call_fn_41075010inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ðBí
*__inference_dropout_layer_call_fn_41075015inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ðBí
*__inference_dropout_layer_call_fn_41075020inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ðBí
*__inference_dropout_layer_call_fn_41075025inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ðBí
*__inference_dropout_layer_call_fn_41075030inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ðBí
*__inference_dropout_layer_call_fn_41075035inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ðBí
*__inference_dropout_layer_call_fn_41075040inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ðBí
*__inference_dropout_layer_call_fn_41075045inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_layer_call_and_return_conditional_losses_41075050inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_layer_call_and_return_conditional_losses_41075055inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_layer_call_and_return_conditional_losses_41075060inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_layer_call_and_return_conditional_losses_41075065inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_layer_call_and_return_conditional_losses_41075070inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_layer_call_and_return_conditional_losses_41075075inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_layer_call_and_return_conditional_losses_41075087inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_layer_call_and_return_conditional_losses_41075099inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_layer_call_and_return_conditional_losses_41075111inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_layer_call_and_return_conditional_losses_41075123inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_layer_call_and_return_conditional_losses_41075135inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_layer_call_and_return_conditional_losses_41075147inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_dense_2_layer_call_fn_41075156inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_dense_2_layer_call_and_return_conditional_losses_41075167inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_dense_3_layer_call_fn_41075176inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_dense_3_layer_call_and_return_conditional_losses_41075187inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_dense_4_layer_call_fn_41075196inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_dense_4_layer_call_and_return_conditional_losses_41075207inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_dense_5_layer_call_fn_41075216inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_dense_5_layer_call_and_return_conditional_losses_41075227inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_dense_6_layer_call_fn_41075236inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_dense_6_layer_call_and_return_conditional_losses_41075247inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_dense_7_layer_call_fn_41075256inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_dense_7_layer_call_and_return_conditional_losses_41075267inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_dense_8_layer_call_fn_41075276inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_dense_8_layer_call_and_return_conditional_losses_41075287inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
D:B2+Adam/tic_tac_toe_neural_net/conv2d/kernel/m
6:42)Adam/tic_tac_toe_neural_net/conv2d/bias/m
D:B27Adam/tic_tac_toe_neural_net/batch_normalization/gamma/m
C:A26Adam/tic_tac_toe_neural_net/batch_normalization/beta/m
G:E2-Adam/tic_tac_toe_neural_net/conv2d_1/kernel/m
8:62+Adam/tic_tac_toe_neural_net/conv2d_1/bias/m
F:D29Adam/tic_tac_toe_neural_net/batch_normalization_1/gamma/m
E:C28Adam/tic_tac_toe_neural_net/batch_normalization_1/beta/m
G:E2-Adam/tic_tac_toe_neural_net/conv2d_2/kernel/m
8:62+Adam/tic_tac_toe_neural_net/conv2d_2/bias/m
F:D29Adam/tic_tac_toe_neural_net/batch_normalization_2/gamma/m
E:C28Adam/tic_tac_toe_neural_net/batch_normalization_2/beta/m
G:E2-Adam/tic_tac_toe_neural_net/conv2d_3/kernel/m
8:62+Adam/tic_tac_toe_neural_net/conv2d_3/bias/m
F:D29Adam/tic_tac_toe_neural_net/batch_normalization_3/gamma/m
E:C28Adam/tic_tac_toe_neural_net/batch_normalization_3/beta/m
<::
2*Adam/tic_tac_toe_neural_net/dense/kernel/m
5:32(Adam/tic_tac_toe_neural_net/dense/bias/m
F:D29Adam/tic_tac_toe_neural_net/batch_normalization_4/gamma/m
E:C28Adam/tic_tac_toe_neural_net/batch_normalization_4/beta/m
>:<
2,Adam/tic_tac_toe_neural_net/dense_1/kernel/m
7:52*Adam/tic_tac_toe_neural_net/dense_1/bias/m
F:D29Adam/tic_tac_toe_neural_net/batch_normalization_5/gamma/m
E:C28Adam/tic_tac_toe_neural_net/batch_normalization_5/beta/m
=:;	22,Adam/tic_tac_toe_neural_net/dense_2/kernel/m
6:422*Adam/tic_tac_toe_neural_net/dense_2/bias/m
<::22,Adam/tic_tac_toe_neural_net/dense_3/kernel/m
6:42*Adam/tic_tac_toe_neural_net/dense_3/bias/m
<::2,Adam/tic_tac_toe_neural_net/dense_4/kernel/m
6:42*Adam/tic_tac_toe_neural_net/dense_4/bias/m
<::	2,Adam/tic_tac_toe_neural_net/dense_5/kernel/m
6:4	2*Adam/tic_tac_toe_neural_net/dense_5/bias/m
=:;	22,Adam/tic_tac_toe_neural_net/dense_6/kernel/m
6:422*Adam/tic_tac_toe_neural_net/dense_6/bias/m
<::2(2,Adam/tic_tac_toe_neural_net/dense_7/kernel/m
6:4(2*Adam/tic_tac_toe_neural_net/dense_7/bias/m
<::(2,Adam/tic_tac_toe_neural_net/dense_8/kernel/m
6:42*Adam/tic_tac_toe_neural_net/dense_8/bias/m
D:B2+Adam/tic_tac_toe_neural_net/conv2d/kernel/v
6:42)Adam/tic_tac_toe_neural_net/conv2d/bias/v
D:B27Adam/tic_tac_toe_neural_net/batch_normalization/gamma/v
C:A26Adam/tic_tac_toe_neural_net/batch_normalization/beta/v
G:E2-Adam/tic_tac_toe_neural_net/conv2d_1/kernel/v
8:62+Adam/tic_tac_toe_neural_net/conv2d_1/bias/v
F:D29Adam/tic_tac_toe_neural_net/batch_normalization_1/gamma/v
E:C28Adam/tic_tac_toe_neural_net/batch_normalization_1/beta/v
G:E2-Adam/tic_tac_toe_neural_net/conv2d_2/kernel/v
8:62+Adam/tic_tac_toe_neural_net/conv2d_2/bias/v
F:D29Adam/tic_tac_toe_neural_net/batch_normalization_2/gamma/v
E:C28Adam/tic_tac_toe_neural_net/batch_normalization_2/beta/v
G:E2-Adam/tic_tac_toe_neural_net/conv2d_3/kernel/v
8:62+Adam/tic_tac_toe_neural_net/conv2d_3/bias/v
F:D29Adam/tic_tac_toe_neural_net/batch_normalization_3/gamma/v
E:C28Adam/tic_tac_toe_neural_net/batch_normalization_3/beta/v
<::
2*Adam/tic_tac_toe_neural_net/dense/kernel/v
5:32(Adam/tic_tac_toe_neural_net/dense/bias/v
F:D29Adam/tic_tac_toe_neural_net/batch_normalization_4/gamma/v
E:C28Adam/tic_tac_toe_neural_net/batch_normalization_4/beta/v
>:<
2,Adam/tic_tac_toe_neural_net/dense_1/kernel/v
7:52*Adam/tic_tac_toe_neural_net/dense_1/bias/v
F:D29Adam/tic_tac_toe_neural_net/batch_normalization_5/gamma/v
E:C28Adam/tic_tac_toe_neural_net/batch_normalization_5/beta/v
=:;	22,Adam/tic_tac_toe_neural_net/dense_2/kernel/v
6:422*Adam/tic_tac_toe_neural_net/dense_2/bias/v
<::22,Adam/tic_tac_toe_neural_net/dense_3/kernel/v
6:42*Adam/tic_tac_toe_neural_net/dense_3/bias/v
<::2,Adam/tic_tac_toe_neural_net/dense_4/kernel/v
6:42*Adam/tic_tac_toe_neural_net/dense_4/bias/v
<::	2,Adam/tic_tac_toe_neural_net/dense_5/kernel/v
6:4	2*Adam/tic_tac_toe_neural_net/dense_5/bias/v
=:;	22,Adam/tic_tac_toe_neural_net/dense_6/kernel/v
6:422*Adam/tic_tac_toe_neural_net/dense_6/bias/v
<::2(2,Adam/tic_tac_toe_neural_net/dense_7/kernel/v
6:4(2*Adam/tic_tac_toe_neural_net/dense_7/bias/v
<::(2,Adam/tic_tac_toe_neural_net/dense_8/kernel/v
6:42*Adam/tic_tac_toe_neural_net/dense_8/bias/vû
#__inference__wrapped_model_41071806Ó2"#$%&'()*+,-./0123456789:;?<>=@AEBDCFGHIJKLMNOPQRS8¢5
.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ
ª "cª`
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ	
.
output_2"
output_2ÿÿÿÿÿÿÿÿÿð
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_41074596*+,-N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ð
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_41074614*+,-N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
8__inference_batch_normalization_1_layer_call_fn_41074565*+,-N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
8__inference_batch_normalization_1_layer_call_fn_41074578*+,-N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_410746770123N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ð
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_410746950123N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
8__inference_batch_normalization_2_layer_call_fn_410746460123N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
8__inference_batch_normalization_2_layer_call_fn_410746590123N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_410747586789N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ð
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_410747766789N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
8__inference_batch_normalization_3_layer_call_fn_410747276789N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
8__inference_batch_normalization_3_layer_call_fn_410747406789N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ»
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_41074852d?<>=4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 »
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_41074886d>?<=4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
8__inference_batch_normalization_4_layer_call_fn_41074819W?<>=4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
8__inference_batch_normalization_4_layer_call_fn_41074832W>?<=4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ»
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_41074951dEBDC4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 »
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_41074985dDEBC4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
8__inference_batch_normalization_5_layer_call_fn_41074918WEBDC4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
8__inference_batch_normalization_5_layer_call_fn_41074931WDEBC4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿî
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_41074515$%&'N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 î
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_41074533$%&'N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
6__inference_batch_normalization_layer_call_fn_41074484$%&'N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
6__inference_batch_normalization_layer_call_fn_41074497$%&'N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸
F__inference_conv2d_1_layer_call_and_return_conditional_losses_41074552n()8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_conv2d_1_layer_call_fn_41074542a()8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ¸
F__inference_conv2d_2_layer_call_and_return_conditional_losses_41074633n./8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_conv2d_2_layer_call_fn_41074623a./8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ¸
F__inference_conv2d_3_layer_call_and_return_conditional_losses_41074714n458¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_conv2d_3_layer_call_fn_41074704a458¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿµ
D__inference_conv2d_layer_call_and_return_conditional_losses_41074471m"#7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_layer_call_fn_41074461`"#7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ§
E__inference_dense_1_layer_call_and_return_conditional_losses_41074905^@A0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_1_layer_call_fn_41074895Q@A0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
E__inference_dense_2_layer_call_and_return_conditional_losses_41075167]FG0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ2
 ~
*__inference_dense_2_layer_call_fn_41075156PFG0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ2¥
E__inference_dense_3_layer_call_and_return_conditional_losses_41075187\HI/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ2
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_3_layer_call_fn_41075176OHI/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ2
ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_dense_4_layer_call_and_return_conditional_losses_41075207\JK/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_4_layer_call_fn_41075196OJK/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_dense_5_layer_call_and_return_conditional_losses_41075227\LM/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ	
 }
*__inference_dense_5_layer_call_fn_41075216OLM/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ	¦
E__inference_dense_6_layer_call_and_return_conditional_losses_41075247]NO0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ2
 ~
*__inference_dense_6_layer_call_fn_41075236PNO0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ2¥
E__inference_dense_7_layer_call_and_return_conditional_losses_41075267\PQ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ2
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 }
*__inference_dense_7_layer_call_fn_41075256OPQ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ2
ª "ÿÿÿÿÿÿÿÿÿ(¥
E__inference_dense_8_layer_call_and_return_conditional_losses_41075287\RS/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ(
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_8_layer_call_fn_41075276ORS/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ(
ª "ÿÿÿÿÿÿÿÿÿ¥
C__inference_dense_layer_call_and_return_conditional_losses_41074806^:;0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
(__inference_dense_layer_call_fn_41074796Q:;0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dropout_layer_call_and_return_conditional_losses_41075050^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 §
E__inference_dropout_layer_call_and_return_conditional_losses_41075055^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¥
E__inference_dropout_layer_call_and_return_conditional_losses_41075060\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ2
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ2
 ¥
E__inference_dropout_layer_call_and_return_conditional_losses_41075065\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¥
E__inference_dropout_layer_call_and_return_conditional_losses_41075070\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¥
E__inference_dropout_layer_call_and_return_conditional_losses_41075075\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ(
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 ¥
E__inference_dropout_layer_call_and_return_conditional_losses_41075087\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ(
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 ¥
E__inference_dropout_layer_call_and_return_conditional_losses_41075099\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ2
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ2
 ¥
E__inference_dropout_layer_call_and_return_conditional_losses_41075111\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¥
E__inference_dropout_layer_call_and_return_conditional_losses_41075123\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 §
E__inference_dropout_layer_call_and_return_conditional_losses_41075135^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 §
E__inference_dropout_layer_call_and_return_conditional_losses_41075147^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dropout_layer_call_fn_41074990O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ(
p 
ª "ÿÿÿÿÿÿÿÿÿ(}
*__inference_dropout_layer_call_fn_41074995O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ(
p
ª "ÿÿÿÿÿÿÿÿÿ(}
*__inference_dropout_layer_call_fn_41075000O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ2
p 
ª "ÿÿÿÿÿÿÿÿÿ2}
*__inference_dropout_layer_call_fn_41075005O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ2
p
ª "ÿÿÿÿÿÿÿÿÿ2}
*__inference_dropout_layer_call_fn_41075010O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ}
*__inference_dropout_layer_call_fn_41075015O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ}
*__inference_dropout_layer_call_fn_41075020O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ}
*__inference_dropout_layer_call_fn_41075025O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_dropout_layer_call_fn_41075030Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_dropout_layer_call_fn_41075035Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_dropout_layer_call_fn_41075040Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_dropout_layer_call_fn_41075045Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ«
E__inference_flatten_layer_call_and_return_conditional_losses_41074787b8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_flatten_layer_call_fn_41074781U8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_signature_wrapper_41073775Þ2"#$%&'()*+,-./0123456789:;?<>=@AEBDCFGHIJKLMNOPQRSC¢@
¢ 
9ª6
4
input_1)&
input_1ÿÿÿÿÿÿÿÿÿ"cª`
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ	
.
output_2"
output_2ÿÿÿÿÿÿÿÿÿ
T__inference_tic_tac_toe_neural_net_layer_call_and_return_conditional_losses_41073528¿2"#$%&'()*+,-./0123456789:;?<>=@AEBDCFGHIJKLMNOPQRS<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "K¢H
A¢>

0/0ÿÿÿÿÿÿÿÿÿ	

0/1ÿÿÿÿÿÿÿÿÿ
 
T__inference_tic_tac_toe_neural_net_layer_call_and_return_conditional_losses_41073666¿2"#$%&'()*+,-./0123456789:;>?<=@ADEBCFGHIJKLMNOPQRS<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "K¢H
A¢>

0/0ÿÿÿÿÿÿÿÿÿ	

0/1ÿÿÿÿÿÿÿÿÿ
 
T__inference_tic_tac_toe_neural_net_layer_call_and_return_conditional_losses_41074182¹2"#$%&'()*+,-./0123456789:;?<>=@AEBDCFGHIJKLMNOPQRS6¢3
,¢)
# 
xÿÿÿÿÿÿÿÿÿ
p 
ª "K¢H
A¢>

0/0ÿÿÿÿÿÿÿÿÿ	

0/1ÿÿÿÿÿÿÿÿÿ
 
T__inference_tic_tac_toe_neural_net_layer_call_and_return_conditional_losses_41074452¹2"#$%&'()*+,-./0123456789:;>?<=@ADEBCFGHIJKLMNOPQRS6¢3
,¢)
# 
xÿÿÿÿÿÿÿÿÿ
p
ª "K¢H
A¢>

0/0ÿÿÿÿÿÿÿÿÿ	

0/1ÿÿÿÿÿÿÿÿÿ
 ï
9__inference_tic_tac_toe_neural_net_layer_call_fn_41072662±2"#$%&'()*+,-./0123456789:;?<>=@AEBDCFGHIJKLMNOPQRS<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "=¢:

0ÿÿÿÿÿÿÿÿÿ	

1ÿÿÿÿÿÿÿÿÿï
9__inference_tic_tac_toe_neural_net_layer_call_fn_41073390±2"#$%&'()*+,-./0123456789:;>?<=@ADEBCFGHIJKLMNOPQRS<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "=¢:

0ÿÿÿÿÿÿÿÿÿ	

1ÿÿÿÿÿÿÿÿÿé
9__inference_tic_tac_toe_neural_net_layer_call_fn_41073882«2"#$%&'()*+,-./0123456789:;?<>=@AEBDCFGHIJKLMNOPQRS6¢3
,¢)
# 
xÿÿÿÿÿÿÿÿÿ
p 
ª "=¢:

0ÿÿÿÿÿÿÿÿÿ	

1ÿÿÿÿÿÿÿÿÿé
9__inference_tic_tac_toe_neural_net_layer_call_fn_41073989«2"#$%&'()*+,-./0123456789:;>?<=@ADEBCFGHIJKLMNOPQRS6¢3
,¢)
# 
xÿÿÿÿÿÿÿÿÿ
p
ª "=¢:

0ÿÿÿÿÿÿÿÿÿ	

1ÿÿÿÿÿÿÿÿÿ