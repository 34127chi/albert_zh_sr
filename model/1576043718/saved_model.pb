��
�!�!
:
Add
x"T
y"T
z"T"
Ttype:
2	
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint�
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
B
AssignVariableOp
resource
value"dtype"
dtypetype�
h
BatchMatMul
x"T
y"T
output"T"
Ttype:
	2"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
�
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint���������"	
Ttype"
TItype0	:
2	
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
6
Pow
x"T
y"T
z"T"
Ttype:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
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
list(type)(0�
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
list(type)(0�
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
1
Square
x"T
y"T"
Ttype:

2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�
9
VarIsInitializedOp
resource
is_initialized
�
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �"serve*1.12.02v1.12.0-0-ga6d8ffae098��

global_step/Initializer/zerosConst*
_class
loc:@global_step*
value	B	 R *
dtype0	*
_output_shapes
: 
�
global_stepVarHandleOp*
shape: *
shared_nameglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
g
,global_step/IsInitialized/VarIsInitializedOpVarIsInitializedOpglobal_step*
_output_shapes
: 

global_step/AssignAssignVariableOpglobal_stepglobal_step/Initializer/zeros*
_class
loc:@global_step*
dtype0	
�
global_step/Read/ReadVariableOpReadVariableOpglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
p
a_input_idsPlaceholder*
dtype0	*(
_output_shapes
:����������*
shape:����������
q
a_input_maskPlaceholder*
shape:����������*
dtype0	*(
_output_shapes
:����������
r
a_segment_idsPlaceholder*
shape:����������*
dtype0	*(
_output_shapes
:����������
p
b_input_idsPlaceholder*
dtype0	*(
_output_shapes
:����������*
shape:����������
q
b_input_maskPlaceholder*
shape:����������*
dtype0	*(
_output_shapes
:����������
r
b_segment_idsPlaceholder*
shape:����������*
dtype0	*(
_output_shapes
:����������
e

unique_idsPlaceholder*
dtype0*#
_output_shapes
:���������*
shape:���������
?
ShapeShape
unique_ids*
T0*
_output_shapes
:
O

ones/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
M
onesFillShape
ones/Const*
T0*#
_output_shapes
:���������
B
Shape_1Shapea_input_ids*
T0	*
_output_shapes
:
]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
strided_sliceStridedSliceShape_1strided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
q
bert/embeddings/ExpandDims/dimConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
bert/embeddings/ExpandDims
ExpandDimsa_input_idsbert/embeddings/ExpandDims/dim*
T0	*,
_output_shapes
:����������
�
Bbert/embeddings/word_embeddings/Initializer/truncated_normal/shapeConst*2
_class(
&$loc:@bert/embeddings/word_embeddings*
valueB"�R  �   *
dtype0*
_output_shapes
:
�
Abert/embeddings/word_embeddings/Initializer/truncated_normal/meanConst*2
_class(
&$loc:@bert/embeddings/word_embeddings*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Cbert/embeddings/word_embeddings/Initializer/truncated_normal/stddevConst*2
_class(
&$loc:@bert/embeddings/word_embeddings*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
�
Lbert/embeddings/word_embeddings/Initializer/truncated_normal/TruncatedNormalTruncatedNormalBbert/embeddings/word_embeddings/Initializer/truncated_normal/shape*
T0*2
_class(
&$loc:@bert/embeddings/word_embeddings*
dtype0*!
_output_shapes
:���
�
@bert/embeddings/word_embeddings/Initializer/truncated_normal/mulMulLbert/embeddings/word_embeddings/Initializer/truncated_normal/TruncatedNormalCbert/embeddings/word_embeddings/Initializer/truncated_normal/stddev*
T0*2
_class(
&$loc:@bert/embeddings/word_embeddings*!
_output_shapes
:���
�
<bert/embeddings/word_embeddings/Initializer/truncated_normalAdd@bert/embeddings/word_embeddings/Initializer/truncated_normal/mulAbert/embeddings/word_embeddings/Initializer/truncated_normal/mean*
T0*2
_class(
&$loc:@bert/embeddings/word_embeddings*!
_output_shapes
:���
�
bert/embeddings/word_embeddings
VariableV2*
shape:���*2
_class(
&$loc:@bert/embeddings/word_embeddings*
dtype0*!
_output_shapes
:���
�
&bert/embeddings/word_embeddings/AssignAssignbert/embeddings/word_embeddings<bert/embeddings/word_embeddings/Initializer/truncated_normal*
T0*2
_class(
&$loc:@bert/embeddings/word_embeddings*!
_output_shapes
:���
�
$bert/embeddings/word_embeddings/readIdentitybert/embeddings/word_embeddings*
T0*2
_class(
&$loc:@bert/embeddings/word_embeddings*!
_output_shapes
:���
�
%bert/embeddings/embedding_lookup/axisConst*2
_class(
&$loc:@bert/embeddings/word_embeddings*
value	B : *
dtype0*
_output_shapes
: 
�
 bert/embeddings/embedding_lookupGatherV2$bert/embeddings/word_embeddings/readbert/embeddings/ExpandDims%bert/embeddings/embedding_lookup/axis*
Taxis0*
Tindices0	*
Tparams0*2
_class(
&$loc:@bert/embeddings/word_embeddings*1
_output_shapes
:�����������
�
)bert/embeddings/embedding_lookup/IdentityIdentity bert/embeddings/embedding_lookup*
T0*1
_output_shapes
:�����������
_
bert/embeddings/ShapeShapebert/embeddings/ExpandDims*
T0	*
_output_shapes
:
m
#bert/embeddings/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
o
%bert/embeddings/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%bert/embeddings/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
bert/embeddings/strided_sliceStridedSlicebert/embeddings/Shape#bert/embeddings/strided_slice/stack%bert/embeddings/strided_slice/stack_1%bert/embeddings/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
b
bert/embeddings/Reshape/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
b
bert/embeddings/Reshape/shape/2Const*
value
B :�*
dtype0*
_output_shapes
: 
�
bert/embeddings/Reshape/shapePackbert/embeddings/strided_slicebert/embeddings/Reshape/shape/1bert/embeddings/Reshape/shape/2*
T0*
N*
_output_shapes
:
�
bert/embeddings/ReshapeReshape)bert/embeddings/embedding_lookup/Identitybert/embeddings/Reshape/shape*
T0*-
_output_shapes
:�����������
^
bert/embeddings/Shape_1Shapebert/embeddings/Reshape*
T0*
_output_shapes
:
o
%bert/embeddings/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
q
'bert/embeddings/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
q
'bert/embeddings/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
bert/embeddings/strided_slice_1StridedSlicebert/embeddings/Shape_1%bert/embeddings/strided_slice_1/stack'bert/embeddings/strided_slice_1/stack_1'bert/embeddings/strided_slice_1/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
�
Hbert/embeddings/token_type_embeddings/Initializer/truncated_normal/shapeConst*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
valueB"   �   *
dtype0*
_output_shapes
:
�
Gbert/embeddings/token_type_embeddings/Initializer/truncated_normal/meanConst*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Ibert/embeddings/token_type_embeddings/Initializer/truncated_normal/stddevConst*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
�
Rbert/embeddings/token_type_embeddings/Initializer/truncated_normal/TruncatedNormalTruncatedNormalHbert/embeddings/token_type_embeddings/Initializer/truncated_normal/shape*
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
dtype0*
_output_shapes
:	�
�
Fbert/embeddings/token_type_embeddings/Initializer/truncated_normal/mulMulRbert/embeddings/token_type_embeddings/Initializer/truncated_normal/TruncatedNormalIbert/embeddings/token_type_embeddings/Initializer/truncated_normal/stddev*
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
_output_shapes
:	�
�
Bbert/embeddings/token_type_embeddings/Initializer/truncated_normalAddFbert/embeddings/token_type_embeddings/Initializer/truncated_normal/mulGbert/embeddings/token_type_embeddings/Initializer/truncated_normal/mean*
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
_output_shapes
:	�
�
%bert/embeddings/token_type_embeddings
VariableV2*
shape:	�*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
dtype0*
_output_shapes
:	�
�
,bert/embeddings/token_type_embeddings/AssignAssign%bert/embeddings/token_type_embeddingsBbert/embeddings/token_type_embeddings/Initializer/truncated_normal*
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
_output_shapes
:	�
�
*bert/embeddings/token_type_embeddings/readIdentity%bert/embeddings/token_type_embeddings*
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
_output_shapes
:	�
r
bert/embeddings/Reshape_1/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
bert/embeddings/Reshape_1Reshapea_segment_idsbert/embeddings/Reshape_1/shape*
T0	*#
_output_shapes
:���������
e
 bert/embeddings/one_hot/on_valueConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
!bert/embeddings/one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_
bert/embeddings/one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 
�
bert/embeddings/one_hotOneHotbert/embeddings/Reshape_1bert/embeddings/one_hot/depth bert/embeddings/one_hot/on_value!bert/embeddings/one_hot/off_value*
T0*'
_output_shapes
:���������
�
bert/embeddings/MatMulMatMulbert/embeddings/one_hot*bert/embeddings/token_type_embeddings/read*
T0*(
_output_shapes
:����������
d
!bert/embeddings/Reshape_2/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
d
!bert/embeddings/Reshape_2/shape/2Const*
value
B :�*
dtype0*
_output_shapes
: 
�
bert/embeddings/Reshape_2/shapePackbert/embeddings/strided_slice_1!bert/embeddings/Reshape_2/shape/1!bert/embeddings/Reshape_2/shape/2*
T0*
N*
_output_shapes
:
�
bert/embeddings/Reshape_2Reshapebert/embeddings/MatMulbert/embeddings/Reshape_2/shape*
T0*-
_output_shapes
:�����������
�
bert/embeddings/addAddbert/embeddings/Reshapebert/embeddings/Reshape_2*
T0*-
_output_shapes
:�����������
f
#bert/embeddings/assert_less_equal/xConst*
value
B :�*
dtype0*
_output_shapes
: 
f
#bert/embeddings/assert_less_equal/yConst*
value
B :�*
dtype0*
_output_shapes
: 
�
+bert/embeddings/assert_less_equal/LessEqual	LessEqual#bert/embeddings/assert_less_equal/x#bert/embeddings/assert_less_equal/y*
T0*
_output_shapes
: 
j
'bert/embeddings/assert_less_equal/ConstConst*
valueB *
dtype0*
_output_shapes
: 
�
%bert/embeddings/assert_less_equal/AllAll+bert/embeddings/assert_less_equal/LessEqual'bert/embeddings/assert_less_equal/Const*
_output_shapes
: 
o
.bert/embeddings/assert_less_equal/Assert/ConstConst*
valueB B *
dtype0*
_output_shapes
: 
�
0bert/embeddings/assert_less_equal/Assert/Const_1Const*h
value_B] BWCondition x <= y did not hold element-wise:x (bert/embeddings/assert_less_equal/x:0) = *
dtype0*
_output_shapes
: 
�
0bert/embeddings/assert_less_equal/Assert/Const_2Const*=
value4B2 B,y (bert/embeddings/assert_less_equal/y:0) = *
dtype0*
_output_shapes
: 
w
6bert/embeddings/assert_less_equal/Assert/Assert/data_0Const*
valueB B *
dtype0*
_output_shapes
: 
�
6bert/embeddings/assert_less_equal/Assert/Assert/data_1Const*h
value_B] BWCondition x <= y did not hold element-wise:x (bert/embeddings/assert_less_equal/x:0) = *
dtype0*
_output_shapes
: 
�
6bert/embeddings/assert_less_equal/Assert/Assert/data_3Const*=
value4B2 B,y (bert/embeddings/assert_less_equal/y:0) = *
dtype0*
_output_shapes
: 
�
/bert/embeddings/assert_less_equal/Assert/AssertAssert%bert/embeddings/assert_less_equal/All6bert/embeddings/assert_less_equal/Assert/Assert/data_06bert/embeddings/assert_less_equal/Assert/Assert/data_1#bert/embeddings/assert_less_equal/x6bert/embeddings/assert_less_equal/Assert/Assert/data_3#bert/embeddings/assert_less_equal/y*
T	
2
�
Fbert/embeddings/position_embeddings/Initializer/truncated_normal/shapeConst*6
_class,
*(loc:@bert/embeddings/position_embeddings*
valueB"   �   *
dtype0*
_output_shapes
:
�
Ebert/embeddings/position_embeddings/Initializer/truncated_normal/meanConst*6
_class,
*(loc:@bert/embeddings/position_embeddings*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Gbert/embeddings/position_embeddings/Initializer/truncated_normal/stddevConst*6
_class,
*(loc:@bert/embeddings/position_embeddings*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
�
Pbert/embeddings/position_embeddings/Initializer/truncated_normal/TruncatedNormalTruncatedNormalFbert/embeddings/position_embeddings/Initializer/truncated_normal/shape*
T0*6
_class,
*(loc:@bert/embeddings/position_embeddings*
dtype0* 
_output_shapes
:
��
�
Dbert/embeddings/position_embeddings/Initializer/truncated_normal/mulMulPbert/embeddings/position_embeddings/Initializer/truncated_normal/TruncatedNormalGbert/embeddings/position_embeddings/Initializer/truncated_normal/stddev*
T0*6
_class,
*(loc:@bert/embeddings/position_embeddings* 
_output_shapes
:
��
�
@bert/embeddings/position_embeddings/Initializer/truncated_normalAddDbert/embeddings/position_embeddings/Initializer/truncated_normal/mulEbert/embeddings/position_embeddings/Initializer/truncated_normal/mean*
T0*6
_class,
*(loc:@bert/embeddings/position_embeddings* 
_output_shapes
:
��
�
#bert/embeddings/position_embeddings
VariableV2*6
_class,
*(loc:@bert/embeddings/position_embeddings*
dtype0* 
_output_shapes
:
��*
shape:
��
�
*bert/embeddings/position_embeddings/AssignAssign#bert/embeddings/position_embeddings@bert/embeddings/position_embeddings/Initializer/truncated_normal*
T0*6
_class,
*(loc:@bert/embeddings/position_embeddings* 
_output_shapes
:
��
�
(bert/embeddings/position_embeddings/readIdentity#bert/embeddings/position_embeddings*
T0*6
_class,
*(loc:@bert/embeddings/position_embeddings* 
_output_shapes
:
��
�
bert/embeddings/Slice/beginConst0^bert/embeddings/assert_less_equal/Assert/Assert*
valueB"        *
dtype0*
_output_shapes
:
�
bert/embeddings/Slice/sizeConst0^bert/embeddings/assert_less_equal/Assert/Assert*
valueB"�   ����*
dtype0*
_output_shapes
:
�
bert/embeddings/SliceSlice(bert/embeddings/position_embeddings/readbert/embeddings/Slice/beginbert/embeddings/Slice/size*
Index0*
T0* 
_output_shapes
:
��
�
bert/embeddings/Reshape_3/shapeConst0^bert/embeddings/assert_less_equal/Assert/Assert*!
valueB"   �   �   *
dtype0*
_output_shapes
:
�
bert/embeddings/Reshape_3Reshapebert/embeddings/Slicebert/embeddings/Reshape_3/shape*
T0*$
_output_shapes
:��
�
bert/embeddings/add_1Addbert/embeddings/addbert/embeddings/Reshape_3*
T0*-
_output_shapes
:�����������
�
0bert/embeddings/LayerNorm/beta/Initializer/zerosConst*1
_class'
%#loc:@bert/embeddings/LayerNorm/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
bert/embeddings/LayerNorm/beta
VariableV2*1
_class'
%#loc:@bert/embeddings/LayerNorm/beta*
dtype0*
_output_shapes	
:�*
shape:�
�
%bert/embeddings/LayerNorm/beta/AssignAssignbert/embeddings/LayerNorm/beta0bert/embeddings/LayerNorm/beta/Initializer/zeros*
T0*1
_class'
%#loc:@bert/embeddings/LayerNorm/beta*
_output_shapes	
:�
�
#bert/embeddings/LayerNorm/beta/readIdentitybert/embeddings/LayerNorm/beta*
T0*1
_class'
%#loc:@bert/embeddings/LayerNorm/beta*
_output_shapes	
:�
�
0bert/embeddings/LayerNorm/gamma/Initializer/onesConst*2
_class(
&$loc:@bert/embeddings/LayerNorm/gamma*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
bert/embeddings/LayerNorm/gamma
VariableV2*2
_class(
&$loc:@bert/embeddings/LayerNorm/gamma*
dtype0*
_output_shapes	
:�*
shape:�
�
&bert/embeddings/LayerNorm/gamma/AssignAssignbert/embeddings/LayerNorm/gamma0bert/embeddings/LayerNorm/gamma/Initializer/ones*
T0*2
_class(
&$loc:@bert/embeddings/LayerNorm/gamma*
_output_shapes	
:�
�
$bert/embeddings/LayerNorm/gamma/readIdentitybert/embeddings/LayerNorm/gamma*
T0*2
_class(
&$loc:@bert/embeddings/LayerNorm/gamma*
_output_shapes	
:�
�
8bert/embeddings/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
&bert/embeddings/LayerNorm/moments/meanMeanbert/embeddings/add_18bert/embeddings/LayerNorm/moments/mean/reduction_indices*
T0*,
_output_shapes
:����������*
	keep_dims(
�
.bert/embeddings/LayerNorm/moments/StopGradientStopGradient&bert/embeddings/LayerNorm/moments/mean*
T0*,
_output_shapes
:����������
�
3bert/embeddings/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/embeddings/add_1.bert/embeddings/LayerNorm/moments/StopGradient*
T0*-
_output_shapes
:�����������
�
<bert/embeddings/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
*bert/embeddings/LayerNorm/moments/varianceMean3bert/embeddings/LayerNorm/moments/SquaredDifference<bert/embeddings/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*
T0*,
_output_shapes
:����������
n
)bert/embeddings/LayerNorm/batchnorm/add/yConst*
valueB
 *̼�+*
dtype0*
_output_shapes
: 
�
'bert/embeddings/LayerNorm/batchnorm/addAdd*bert/embeddings/LayerNorm/moments/variance)bert/embeddings/LayerNorm/batchnorm/add/y*
T0*,
_output_shapes
:����������
�
)bert/embeddings/LayerNorm/batchnorm/RsqrtRsqrt'bert/embeddings/LayerNorm/batchnorm/add*
T0*,
_output_shapes
:����������
�
'bert/embeddings/LayerNorm/batchnorm/mulMul)bert/embeddings/LayerNorm/batchnorm/Rsqrt$bert/embeddings/LayerNorm/gamma/read*
T0*-
_output_shapes
:�����������
�
)bert/embeddings/LayerNorm/batchnorm/mul_1Mulbert/embeddings/add_1'bert/embeddings/LayerNorm/batchnorm/mul*
T0*-
_output_shapes
:�����������
�
)bert/embeddings/LayerNorm/batchnorm/mul_2Mul&bert/embeddings/LayerNorm/moments/mean'bert/embeddings/LayerNorm/batchnorm/mul*
T0*-
_output_shapes
:�����������
�
'bert/embeddings/LayerNorm/batchnorm/subSub#bert/embeddings/LayerNorm/beta/read)bert/embeddings/LayerNorm/batchnorm/mul_2*
T0*-
_output_shapes
:�����������
�
)bert/embeddings/LayerNorm/batchnorm/add_1Add)bert/embeddings/LayerNorm/batchnorm/mul_1'bert/embeddings/LayerNorm/batchnorm/sub*
T0*-
_output_shapes
:�����������
k
bert/encoder/ShapeShape)bert/embeddings/LayerNorm/batchnorm/add_1*
T0*
_output_shapes
:
j
 bert/encoder/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
l
"bert/encoder/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
l
"bert/encoder/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
bert/encoder/strided_sliceStridedSlicebert/encoder/Shape bert/encoder/strided_slice/stack"bert/encoder/strided_slice/stack_1"bert/encoder/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
m
bert/encoder/Shape_1Shape)bert/embeddings/LayerNorm/batchnorm/add_1*
T0*
_output_shapes
:
l
"bert/encoder/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
n
$bert/encoder/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
n
$bert/encoder/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
bert/encoder/strided_slice_1StridedSlicebert/encoder/Shape_1"bert/encoder/strided_slice_1/stack$bert/encoder/strided_slice_1/stack_1$bert/encoder/strided_slice_1/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
�
Rbert/encoder/embedding_hidden_mapping_in/kernel/Initializer/truncated_normal/shapeConst*B
_class8
64loc:@bert/encoder/embedding_hidden_mapping_in/kernel*
valueB"�   8  *
dtype0*
_output_shapes
:
�
Qbert/encoder/embedding_hidden_mapping_in/kernel/Initializer/truncated_normal/meanConst*B
_class8
64loc:@bert/encoder/embedding_hidden_mapping_in/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Sbert/encoder/embedding_hidden_mapping_in/kernel/Initializer/truncated_normal/stddevConst*B
_class8
64loc:@bert/encoder/embedding_hidden_mapping_in/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
�
\bert/encoder/embedding_hidden_mapping_in/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalRbert/encoder/embedding_hidden_mapping_in/kernel/Initializer/truncated_normal/shape*
T0*B
_class8
64loc:@bert/encoder/embedding_hidden_mapping_in/kernel*
dtype0* 
_output_shapes
:
��
�
Pbert/encoder/embedding_hidden_mapping_in/kernel/Initializer/truncated_normal/mulMul\bert/encoder/embedding_hidden_mapping_in/kernel/Initializer/truncated_normal/TruncatedNormalSbert/encoder/embedding_hidden_mapping_in/kernel/Initializer/truncated_normal/stddev*
T0*B
_class8
64loc:@bert/encoder/embedding_hidden_mapping_in/kernel* 
_output_shapes
:
��
�
Lbert/encoder/embedding_hidden_mapping_in/kernel/Initializer/truncated_normalAddPbert/encoder/embedding_hidden_mapping_in/kernel/Initializer/truncated_normal/mulQbert/encoder/embedding_hidden_mapping_in/kernel/Initializer/truncated_normal/mean*
T0*B
_class8
64loc:@bert/encoder/embedding_hidden_mapping_in/kernel* 
_output_shapes
:
��
�
/bert/encoder/embedding_hidden_mapping_in/kernel
VariableV2*B
_class8
64loc:@bert/encoder/embedding_hidden_mapping_in/kernel*
dtype0* 
_output_shapes
:
��*
shape:
��
�
6bert/encoder/embedding_hidden_mapping_in/kernel/AssignAssign/bert/encoder/embedding_hidden_mapping_in/kernelLbert/encoder/embedding_hidden_mapping_in/kernel/Initializer/truncated_normal*
T0*B
_class8
64loc:@bert/encoder/embedding_hidden_mapping_in/kernel* 
_output_shapes
:
��
�
4bert/encoder/embedding_hidden_mapping_in/kernel/readIdentity/bert/encoder/embedding_hidden_mapping_in/kernel*
T0*B
_class8
64loc:@bert/encoder/embedding_hidden_mapping_in/kernel* 
_output_shapes
:
��
�
?bert/encoder/embedding_hidden_mapping_in/bias/Initializer/zerosConst*@
_class6
42loc:@bert/encoder/embedding_hidden_mapping_in/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
-bert/encoder/embedding_hidden_mapping_in/bias
VariableV2*@
_class6
42loc:@bert/encoder/embedding_hidden_mapping_in/bias*
dtype0*
_output_shapes	
:�*
shape:�
�
4bert/encoder/embedding_hidden_mapping_in/bias/AssignAssign-bert/encoder/embedding_hidden_mapping_in/bias?bert/encoder/embedding_hidden_mapping_in/bias/Initializer/zeros*
T0*@
_class6
42loc:@bert/encoder/embedding_hidden_mapping_in/bias*
_output_shapes	
:�
�
2bert/encoder/embedding_hidden_mapping_in/bias/readIdentity-bert/encoder/embedding_hidden_mapping_in/bias*
T0*@
_class6
42loc:@bert/encoder/embedding_hidden_mapping_in/bias*
_output_shapes	
:�
�
>bert/encoder/embedding_hidden_mapping_in/einsum/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
9bert/encoder/embedding_hidden_mapping_in/einsum/transpose	Transpose)bert/embeddings/LayerNorm/batchnorm/add_1>bert/encoder/embedding_hidden_mapping_in/einsum/transpose/perm*
T0*-
_output_shapes
:�����������
�
@bert/encoder/embedding_hidden_mapping_in/einsum/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
;bert/encoder/embedding_hidden_mapping_in/einsum/transpose_1	Transpose4bert/encoder/embedding_hidden_mapping_in/kernel/read@bert/encoder/embedding_hidden_mapping_in/einsum/transpose_1/perm*
T0* 
_output_shapes
:
��
�
5bert/encoder/embedding_hidden_mapping_in/einsum/ShapeShape9bert/encoder/embedding_hidden_mapping_in/einsum/transpose*
T0*
_output_shapes
:
�
Cbert/encoder/embedding_hidden_mapping_in/einsum/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Ebert/encoder/embedding_hidden_mapping_in/einsum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Ebert/encoder/embedding_hidden_mapping_in/einsum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
=bert/encoder/embedding_hidden_mapping_in/einsum/strided_sliceStridedSlice5bert/encoder/embedding_hidden_mapping_in/einsum/ShapeCbert/encoder/embedding_hidden_mapping_in/einsum/strided_slice/stackEbert/encoder/embedding_hidden_mapping_in/einsum/strided_slice/stack_1Ebert/encoder/embedding_hidden_mapping_in/einsum/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
w
5bert/encoder/embedding_hidden_mapping_in/einsum/mul/xConst*
value	B :*
dtype0*
_output_shapes
: 
�
3bert/encoder/embedding_hidden_mapping_in/einsum/mulMul5bert/encoder/embedding_hidden_mapping_in/einsum/mul/x=bert/encoder/embedding_hidden_mapping_in/einsum/strided_slice*
T0*
_output_shapes
: 
z
7bert/encoder/embedding_hidden_mapping_in/einsum/mul_1/yConst*
value
B :�*
dtype0*
_output_shapes
: 
�
5bert/encoder/embedding_hidden_mapping_in/einsum/mul_1Mul3bert/encoder/embedding_hidden_mapping_in/einsum/mul7bert/encoder/embedding_hidden_mapping_in/einsum/mul_1/y*
T0*
_output_shapes
: 
�
?bert/encoder/embedding_hidden_mapping_in/einsum/Reshape/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
=bert/encoder/embedding_hidden_mapping_in/einsum/Reshape/shapePack5bert/encoder/embedding_hidden_mapping_in/einsum/mul_1?bert/encoder/embedding_hidden_mapping_in/einsum/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
7bert/encoder/embedding_hidden_mapping_in/einsum/ReshapeReshape9bert/encoder/embedding_hidden_mapping_in/einsum/transpose=bert/encoder/embedding_hidden_mapping_in/einsum/Reshape/shape*
T0*(
_output_shapes
:����������
�
6bert/encoder/embedding_hidden_mapping_in/einsum/MatMulMatMul7bert/encoder/embedding_hidden_mapping_in/einsum/Reshape;bert/encoder/embedding_hidden_mapping_in/einsum/transpose_1*
T0*(
_output_shapes
:����������
�
Abert/encoder/embedding_hidden_mapping_in/einsum/Reshape_1/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
Abert/encoder/embedding_hidden_mapping_in/einsum/Reshape_1/shape/2Const*
value
B :�*
dtype0*
_output_shapes
: 
�
?bert/encoder/embedding_hidden_mapping_in/einsum/Reshape_1/shapePack=bert/encoder/embedding_hidden_mapping_in/einsum/strided_sliceAbert/encoder/embedding_hidden_mapping_in/einsum/Reshape_1/shape/1Abert/encoder/embedding_hidden_mapping_in/einsum/Reshape_1/shape/2*
T0*
N*
_output_shapes
:
�
9bert/encoder/embedding_hidden_mapping_in/einsum/Reshape_1Reshape6bert/encoder/embedding_hidden_mapping_in/einsum/MatMul?bert/encoder/embedding_hidden_mapping_in/einsum/Reshape_1/shape*
T0*-
_output_shapes
:�����������
�
@bert/encoder/embedding_hidden_mapping_in/einsum/transpose_2/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
;bert/encoder/embedding_hidden_mapping_in/einsum/transpose_2	Transpose9bert/encoder/embedding_hidden_mapping_in/einsum/Reshape_1@bert/encoder/embedding_hidden_mapping_in/einsum/transpose_2/perm*
T0*-
_output_shapes
:�����������
�
,bert/encoder/embedding_hidden_mapping_in/addAdd;bert/encoder/embedding_hidden_mapping_in/einsum/transpose_22bert/encoder/embedding_hidden_mapping_in/bias/read*
T0*-
_output_shapes
:�����������
�
Mbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ShapeShape,bert/encoder/embedding_hidden_mapping_in/add*
T0*
_output_shapes
:
�
[bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Ubert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_sliceStridedSliceMbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape[bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice/stack]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice/stack_1]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
�
Obert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_1Shape,bert/encoder/embedding_hidden_mapping_in/add*
T0*
_output_shapes
:
�
]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Wbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_1StridedSliceObert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_1]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_1/stack_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_1/stack_1_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_1/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
�
Obert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_2Shape,bert/encoder/embedding_hidden_mapping_in/add*
T0*
_output_shapes
:
�
]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Wbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_2StridedSliceObert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_2]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_2/stack_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_2/stack_1_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_2/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
�
obert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel/Initializer/truncated_normal/shapeConst*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel*
valueB"8  8  *
dtype0*
_output_shapes
:
�
nbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel/Initializer/truncated_normal/meanConst*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
pbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel/Initializer/truncated_normal/stddevConst*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
�
ybert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalobert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel/Initializer/truncated_normal/shape*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel*
dtype0* 
_output_shapes
:
��
�
mbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel/Initializer/truncated_normal/mulMulybert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel/Initializer/truncated_normal/TruncatedNormalpbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel/Initializer/truncated_normal/stddev*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel* 
_output_shapes
:
��
�
ibert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel/Initializer/truncated_normalAddmbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel/Initializer/truncated_normal/mulnbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel/Initializer/truncated_normal/mean*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel* 
_output_shapes
:
��
�
Lbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel
VariableV2*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel*
dtype0* 
_output_shapes
:
��*
shape:
��
�
Sbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel/AssignAssignLbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernelibert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel/Initializer/truncated_normal*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel* 
_output_shapes
:
��
�
Qbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel/readIdentityLbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel* 
_output_shapes
:
��
�
[bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/Reshape/shapeConst*!
valueB"8        *
dtype0*
_output_shapes
:
�
Ubert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/ReshapeReshapeQbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel/read[bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/Reshape/shape*
T0*#
_output_shapes
:�
�
\bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias/Initializer/zerosConst*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Jbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias
VariableV2*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias*
dtype0*
_output_shapes	
:�*
shape:�
�
Qbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias/AssignAssignJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias\bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias/Initializer/zeros*
T0*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias*
_output_shapes	
:�
�
Obert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias/readIdentityJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias*
T0*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias*
_output_shapes	
:�
�
]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/Reshape_1/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
Wbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/Reshape_1ReshapeObert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias/read]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/Reshape_1/shape*
T0*
_output_shapes

:
�
cbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
^bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose	Transpose,bert/encoder/embedding_hidden_mapping_in/addcbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose/perm*
T0*-
_output_shapes
:�����������
�
ebert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
`bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose_1	TransposeUbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/Reshapeebert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose_1/perm*
T0*#
_output_shapes
:�
�
Zbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/ShapeShape^bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose*
T0*
_output_shapes
:
�
hbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
jbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
jbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
bbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/strided_sliceStridedSliceZbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Shapehbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/strided_slice/stackjbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/strided_slice/stack_1jbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
�
Zbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/mul/xConst*
value	B :*
dtype0*
_output_shapes
: 
�
Xbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/mulMulZbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/mul/xbbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/strided_slice*
T0*
_output_shapes
: 
�
\bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/mul_1/yConst*
value
B :�*
dtype0*
_output_shapes
: 
�
Zbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/mul_1MulXbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/mul\bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/mul_1/y*
T0*
_output_shapes
: 
�
dbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
bbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape/shapePackZbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/mul_1dbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
\bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/ReshapeReshape^bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transposebbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape/shape*
T0*(
_output_shapes
:����������
�
dbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_1/shapeConst*
valueB"8  8  *
dtype0*
_output_shapes
:
�
^bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_1Reshape`bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose_1dbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_1/shape*
T0* 
_output_shapes
:
��
�
[bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/MatMulMatMul\bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape^bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_1*
T0*(
_output_shapes
:����������
�
fbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_2/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
fbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_2/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
�
fbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_2/shape/3Const*
value	B :*
dtype0*
_output_shapes
: 
�
dbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_2/shapePackbbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/strided_slicefbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_2/shape/1fbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_2/shape/2fbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_2/shape/3*
T0*
N*
_output_shapes
:
�
^bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_2Reshape[bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/MatMuldbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_2/shape*
T0*0
_output_shapes
:����������
�
ebert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
�
`bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose_2	Transpose^bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_2ebert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose_2/perm*
T0*0
_output_shapes
:����������
�
Qbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/addAdd`bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose_2Wbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/Reshape_1*
T0*0
_output_shapes
:����������
�
Obert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_3Shape,bert/encoder/embedding_hidden_mapping_in/add*
T0*
_output_shapes
:
�
]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Wbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_3StridedSliceObert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_3]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_3/stack_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_3/stack_1_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_3/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
�
mbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel/Initializer/truncated_normal/shapeConst*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel*
valueB"8  8  *
dtype0*
_output_shapes
:
�
lbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel/Initializer/truncated_normal/meanConst*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
nbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel/Initializer/truncated_normal/stddevConst*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
�
wbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalmbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel/Initializer/truncated_normal/shape*
T0*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel*
dtype0* 
_output_shapes
:
��
�
kbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel/Initializer/truncated_normal/mulMulwbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel/Initializer/truncated_normal/TruncatedNormalnbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel/Initializer/truncated_normal/stddev*
T0*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel* 
_output_shapes
:
��
�
gbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel/Initializer/truncated_normalAddkbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel/Initializer/truncated_normal/mullbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel/Initializer/truncated_normal/mean*
T0*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel* 
_output_shapes
:
��
�
Jbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel
VariableV2*
shape:
��*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel*
dtype0* 
_output_shapes
:
��
�
Qbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel/AssignAssignJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernelgbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel/Initializer/truncated_normal*
T0*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel* 
_output_shapes
:
��
�
Obert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel/readIdentityJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel*
T0*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel* 
_output_shapes
:
��
�
Ybert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/Reshape/shapeConst*!
valueB"8        *
dtype0*
_output_shapes
:
�
Sbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/ReshapeReshapeObert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel/readYbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/Reshape/shape*
T0*#
_output_shapes
:�
�
Zbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias/Initializer/zerosConst*[
_classQ
OMloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias
VariableV2*
shape:�*[
_classQ
OMloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias*
dtype0*
_output_shapes	
:�
�
Obert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias/AssignAssignHbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/biasZbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias/Initializer/zeros*
T0*[
_classQ
OMloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias*
_output_shapes	
:�
�
Mbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias/readIdentityHbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias*
T0*[
_classQ
OMloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias*
_output_shapes	
:�
�
[bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/Reshape_1/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
Ubert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/Reshape_1ReshapeMbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias/read[bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/Reshape_1/shape*
T0*
_output_shapes

:
�
abert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
\bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose	Transpose,bert/encoder/embedding_hidden_mapping_in/addabert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose/perm*
T0*-
_output_shapes
:�����������
�
cbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
^bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose_1	TransposeSbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/Reshapecbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose_1/perm*
T0*#
_output_shapes
:�
�
Xbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/ShapeShape\bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose*
T0*
_output_shapes
:
�
fbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
hbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
hbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
`bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/strided_sliceStridedSliceXbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Shapefbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/strided_slice/stackhbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/strided_slice/stack_1hbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
�
Xbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/mul/xConst*
value	B :*
dtype0*
_output_shapes
: 
�
Vbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/mulMulXbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/mul/x`bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/strided_slice*
T0*
_output_shapes
: 
�
Zbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/mul_1/yConst*
value
B :�*
dtype0*
_output_shapes
: 
�
Xbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/mul_1MulVbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/mulZbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/mul_1/y*
T0*
_output_shapes
: 
�
bbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
`bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape/shapePackXbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/mul_1bbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
Zbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/ReshapeReshape\bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose`bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape/shape*
T0*(
_output_shapes
:����������
�
bbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_1/shapeConst*
valueB"8  8  *
dtype0*
_output_shapes
:
�
\bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_1Reshape^bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose_1bbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_1/shape*
T0* 
_output_shapes
:
��
�
Ybert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/MatMulMatMulZbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape\bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_1*
T0*(
_output_shapes
:����������
�
dbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_2/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
dbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_2/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
�
dbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_2/shape/3Const*
value	B :*
dtype0*
_output_shapes
: 
�
bbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_2/shapePack`bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/strided_slicedbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_2/shape/1dbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_2/shape/2dbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_2/shape/3*
T0*
N*
_output_shapes
:
�
\bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_2ReshapeYbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/MatMulbbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_2/shape*
T0*0
_output_shapes
:����������
�
cbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
�
^bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose_2	Transpose\bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_2cbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose_2/perm*
T0*0
_output_shapes
:����������
�
Obert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/addAdd^bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose_2Ubert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/Reshape_1*
T0*0
_output_shapes
:����������
�
Obert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_4Shape,bert/encoder/embedding_hidden_mapping_in/add*
T0*
_output_shapes
:
�
]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_4/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_4/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Wbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_4StridedSliceObert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_4]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_4/stack_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_4/stack_1_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_4/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
�
obert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel/Initializer/truncated_normal/shapeConst*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel*
valueB"8  8  *
dtype0*
_output_shapes
:
�
nbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel/Initializer/truncated_normal/meanConst*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
pbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel/Initializer/truncated_normal/stddevConst*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
�
ybert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalobert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel/Initializer/truncated_normal/shape*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel*
dtype0* 
_output_shapes
:
��
�
mbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel/Initializer/truncated_normal/mulMulybert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel/Initializer/truncated_normal/TruncatedNormalpbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel/Initializer/truncated_normal/stddev*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel* 
_output_shapes
:
��
�
ibert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel/Initializer/truncated_normalAddmbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel/Initializer/truncated_normal/mulnbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel/Initializer/truncated_normal/mean*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel* 
_output_shapes
:
��
�
Lbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel
VariableV2*
shape:
��*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel*
dtype0* 
_output_shapes
:
��
�
Sbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel/AssignAssignLbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernelibert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel/Initializer/truncated_normal*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel* 
_output_shapes
:
��
�
Qbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel/readIdentityLbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel* 
_output_shapes
:
��
�
[bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/Reshape/shapeConst*!
valueB"8        *
dtype0*
_output_shapes
:
�
Ubert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/ReshapeReshapeQbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel/read[bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/Reshape/shape*
T0*#
_output_shapes
:�
�
\bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias/Initializer/zerosConst*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Jbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias
VariableV2*
shape:�*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias*
dtype0*
_output_shapes	
:�
�
Qbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias/AssignAssignJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias\bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias/Initializer/zeros*
T0*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias*
_output_shapes	
:�
�
Obert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias/readIdentityJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias*
T0*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias*
_output_shapes	
:�
�
]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/Reshape_1/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
Wbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/Reshape_1ReshapeObert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias/read]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/Reshape_1/shape*
T0*
_output_shapes

:
�
cbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
^bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose	Transpose,bert/encoder/embedding_hidden_mapping_in/addcbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose/perm*
T0*-
_output_shapes
:�����������
�
ebert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
`bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose_1	TransposeUbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/Reshapeebert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose_1/perm*
T0*#
_output_shapes
:�
�
Zbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/ShapeShape^bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose*
T0*
_output_shapes
:
�
hbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
jbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
jbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
bbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/strided_sliceStridedSliceZbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Shapehbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/strided_slice/stackjbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/strided_slice/stack_1jbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
�
Zbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/mul/xConst*
value	B :*
dtype0*
_output_shapes
: 
�
Xbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/mulMulZbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/mul/xbbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/strided_slice*
T0*
_output_shapes
: 
�
\bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/mul_1/yConst*
value
B :�*
dtype0*
_output_shapes
: 
�
Zbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/mul_1MulXbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/mul\bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/mul_1/y*
T0*
_output_shapes
: 
�
dbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
bbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape/shapePackZbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/mul_1dbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
\bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/ReshapeReshape^bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transposebbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape/shape*
T0*(
_output_shapes
:����������
�
dbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_1/shapeConst*
valueB"8  8  *
dtype0*
_output_shapes
:
�
^bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_1Reshape`bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose_1dbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_1/shape*
T0* 
_output_shapes
:
��
�
[bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/MatMulMatMul\bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape^bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_1*
T0*(
_output_shapes
:����������
�
fbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_2/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
fbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_2/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
�
fbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_2/shape/3Const*
value	B :*
dtype0*
_output_shapes
: 
�
dbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_2/shapePackbbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/strided_slicefbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_2/shape/1fbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_2/shape/2fbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_2/shape/3*
T0*
N*
_output_shapes
:
�
^bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_2Reshape[bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/MatMuldbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_2/shape*
T0*0
_output_shapes
:����������
�
ebert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
�
`bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose_2	Transpose^bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_2ebert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose_2/perm*
T0*0
_output_shapes
:����������
�
Qbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/addAdd`bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose_2Wbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/Reshape_1*
T0*0
_output_shapes
:����������
�
Vbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
�
Qbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose	TransposeQbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/addVbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose/perm*
T0*0
_output_shapes
:����������
�
Xbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
�
Sbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_1	TransposeObert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/addXbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_1/perm*
T0*0
_output_shapes
:����������
�
Xbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
�
Sbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_2	TransposeQbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/addXbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_2/perm*
T0*0
_output_shapes
:����������
�
Wbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Wbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Reshape/shape/2Const*
value
B :�*
dtype0*
_output_shapes
: 
�
Wbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Reshape/shape/3Const*
value	B :*
dtype0*
_output_shapes
: 
�
Ubert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Reshape/shapePackUbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_sliceWbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Reshape/shape/1Wbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Reshape/shape/2Wbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Reshape/shape/3*
T0*
N*
_output_shapes
:
�
Obert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ReshapeReshapea_input_maskUbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Reshape/shape*
T0	*0
_output_shapes
:����������
�
Nbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/MatMulBatchMatMulQbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transposeSbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_1*
T0*1
_output_shapes
:�����������*
adj_y(
�
Obert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_5ShapeQbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose*
T0*
_output_shapes
:
�
]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_5/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_5/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_5/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Wbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_5StridedSliceObert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_5]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_5/stack_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_5/stack_1_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_5/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
�
Mbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Mul/yConst*
valueB
 *��H>*
dtype0*
_output_shapes
: 
�
Kbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/MulMulNbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/MatMulMbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Mul/y*
T0*1
_output_shapes
:�����������
�
Obert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_6ShapeQbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose*
T0*
_output_shapes
:
�
]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_6/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_6/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_6/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Wbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_6StridedSliceObert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_6]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_6/stack_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_6/stack_1_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_6/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
�
Rbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
Pbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mulMulWbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_6Rbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mul/y*
T0*
_output_shapes
: 
�
Tbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mul_1/yConst*
value
B :�*
dtype0*
_output_shapes
: 
�
Rbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mul_1MulPbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mulTbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mul_1/y*
T0*
_output_shapes
: 
�
Tbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mul_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
Rbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mul_2MulRbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mul_1Tbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mul_2/y*
T0*
_output_shapes
: 
�
Sbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/Less/yConst*
value
B :�*
dtype0*
_output_shapes
: 
�
Qbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/LessLessRbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mul_2Sbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/Less/y*
T0*
_output_shapes
: 
�
Ubert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/packed/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Ubert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/packed/2Const*
value
B :�*
dtype0*
_output_shapes
: 
�
Ubert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/packed/3Const*
value	B :*
dtype0*
_output_shapes
: 
�
Sbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/packedPackWbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_6Ubert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/packed/1Ubert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/packed/2Ubert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/packed/3*
T0*
N*
_output_shapes
:
�
Rbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Lbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/onesFillSbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/packedRbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/Const*
T0*0
_output_shapes
:����������
�
Lbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/CastCastObert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Reshape*

SrcT0	*0
_output_shapes
:����������*

DstT0
�
Pbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/MatMul_1BatchMatMulLbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/onesLbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Cast*
adj_y(*
T0*1
_output_shapes
:�����������
�
Mbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Kbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/subSubMbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/sub/xPbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/MatMul_1*
T0*1
_output_shapes
:�����������
�
Obert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/mul_1/yConst*
valueB
 * @�*
dtype0*
_output_shapes
: 
�
Mbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/mul_1MulKbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/subObert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/mul_1/y*
T0*1
_output_shapes
:�����������
�
Kbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/addAddKbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/MulMbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/mul_1*
T0*1
_output_shapes
:�����������
�
Wbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/attention_probsSoftmaxKbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/add*
T0*1
_output_shapes
:�����������
�
Pbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/MatMul_2BatchMatMulWbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/attention_probsSbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_2*
T0*0
_output_shapes
:����������
�
Xbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:
�
Sbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_3	TransposePbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/MatMul_2Xbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_3/perm*
T0*0
_output_shapes
:����������
�
Obert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/ShapeShapeSbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_3*
T0*
_output_shapes
:
�
]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Wbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/strided_sliceStridedSliceObert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/Shape]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/strided_slice/stack_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/strided_slice/stack_1_bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
�
qbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel/Initializer/truncated_normal/shapeConst*a
_classW
USloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel*
valueB"8  8  *
dtype0*
_output_shapes
:
�
pbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel/Initializer/truncated_normal/meanConst*a
_classW
USloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
rbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel/Initializer/truncated_normal/stddevConst*a
_classW
USloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
�
{bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalqbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel/Initializer/truncated_normal/shape*
T0*a
_classW
USloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel*
dtype0* 
_output_shapes
:
��
�
obert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel/Initializer/truncated_normal/mulMul{bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalrbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*a
_classW
USloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel* 
_output_shapes
:
��
�
kbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel/Initializer/truncated_normalAddobert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel/Initializer/truncated_normal/mulpbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel/Initializer/truncated_normal/mean*
T0*a
_classW
USloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel* 
_output_shapes
:
��
�
Nbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel
VariableV2*
shape:
��*a
_classW
USloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel*
dtype0* 
_output_shapes
:
��
�
Ubert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel/AssignAssignNbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernelkbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel/Initializer/truncated_normal*
T0*a
_classW
USloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel* 
_output_shapes
:
��
�
Sbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel/readIdentityNbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel*
T0*a
_classW
USloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel* 
_output_shapes
:
��
�
]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/Reshape/shapeConst*!
valueB"      8  *
dtype0*
_output_shapes
:
�
Wbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/ReshapeReshapeSbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel/read]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/Reshape/shape*
T0*#
_output_shapes
:�
�
^bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias/Initializer/zerosConst*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Lbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias
VariableV2*
shape:�*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias*
dtype0*
_output_shapes	
:�
�
Sbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias/AssignAssignLbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias^bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias/Initializer/zeros*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias*
_output_shapes	
:�
�
Qbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias/readIdentityLbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias*
_output_shapes	
:�
�
ebert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
�
`bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose	TransposeSbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_3ebert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose/perm*
T0*0
_output_shapes
:����������
�
gbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
bbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose_1	TransposeWbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/Reshapegbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose_1/perm*
T0*#
_output_shapes
:�
�
\bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/ShapeShape`bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose*
T0*
_output_shapes
:
�
jbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
lbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
lbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
dbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/strided_sliceStridedSlice\bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Shapejbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/strided_slice/stacklbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/strided_slice/stack_1lbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
�
\bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/mul/xConst*
value	B :*
dtype0*
_output_shapes
: 
�
Zbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/mulMul\bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/mul/xdbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/strided_slice*
T0*
_output_shapes
: 
�
^bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/mul_1/yConst*
value
B :�*
dtype0*
_output_shapes
: 
�
\bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/mul_1MulZbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/mul^bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/mul_1/y*
T0*
_output_shapes
: 
�
fbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
dbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape/shapePack\bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/mul_1fbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
^bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/ReshapeReshape`bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transposedbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape/shape*
T0*(
_output_shapes
:����������
�
fbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_1/shapeConst*
valueB"8  8  *
dtype0*
_output_shapes
:
�
`bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_1Reshapebbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose_1fbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_1/shape*
T0* 
_output_shapes
:
��
�
]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/MatMulMatMul^bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape`bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_1*
T0*(
_output_shapes
:����������
�
hbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_2/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
hbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_2/shape/2Const*
value
B :�*
dtype0*
_output_shapes
: 
�
fbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_2/shapePackdbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/strided_slicehbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_2/shape/1hbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_2/shape/2*
T0*
N*
_output_shapes
:
�
`bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_2Reshape]bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/MatMulfbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_2/shape*
T0*-
_output_shapes
:�����������
�
gbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose_2/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
bbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose_2	Transpose`bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_2gbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose_2/perm*
T0*-
_output_shapes
:�����������
�
Sbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/addAddbbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose_2Qbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias/read*
T0*-
_output_shapes
:�����������
�
:bert/encoder/transformer/group_0/layer_0/inner_group_0/addAddSbert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/add,bert/encoder/embedding_hidden_mapping_in/add*
T0*-
_output_shapes
:�����������
�
Obert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta/Initializer/zerosConst*P
_classF
DBloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
=bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta
VariableV2*P
_classF
DBloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta*
dtype0*
_output_shapes	
:�*
shape:�
�
Dbert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta/AssignAssign=bert/encoder/transformer/group_0/inner_group_0/LayerNorm/betaObert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta/Initializer/zeros*
T0*P
_classF
DBloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta*
_output_shapes	
:�
�
Bbert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta/readIdentity=bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta*
T0*P
_classF
DBloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta*
_output_shapes	
:�
�
Obert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
>bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma
VariableV2*Q
_classG
ECloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma*
dtype0*
_output_shapes	
:�*
shape:�
�
Ebert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma/AssignAssign>bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gammaObert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma/Initializer/ones*
T0*Q
_classG
ECloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma*
_output_shapes	
:�
�
Cbert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma/readIdentity>bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma*
T0*Q
_classG
ECloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma*
_output_shapes	
:�
�
_bert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
Mbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/meanMean:bert/encoder/transformer/group_0/layer_0/inner_group_0/add_bert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*
T0*,
_output_shapes
:����������
�
Ubert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/StopGradientStopGradientMbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/mean*
T0*,
_output_shapes
:����������
�
Zbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/SquaredDifferenceSquaredDifference:bert/encoder/transformer/group_0/layer_0/inner_group_0/addUbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/StopGradient*
T0*-
_output_shapes
:�����������
�
cbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
Qbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/varianceMeanZbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/SquaredDifferencecbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/variance/reduction_indices*
T0*,
_output_shapes
:����������*
	keep_dims(
�
Pbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/add/yConst*
valueB
 *̼�+*
dtype0*
_output_shapes
: 
�
Nbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/addAddQbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/variancePbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/add/y*
T0*,
_output_shapes
:����������
�
Pbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/RsqrtRsqrtNbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/add*
T0*,
_output_shapes
:����������
�
Nbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/mulMulPbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/RsqrtCbert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma/read*
T0*-
_output_shapes
:�����������
�
Pbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/mul_1Mul:bert/encoder/transformer/group_0/layer_0/inner_group_0/addNbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/mul*
T0*-
_output_shapes
:�����������
�
Pbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/mul_2MulMbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/meanNbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/mul*
T0*-
_output_shapes
:�����������
�
Nbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/subSubBbert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta/readPbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/mul_2*
T0*-
_output_shapes
:�����������
�
Pbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/add_1AddPbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/mul_1Nbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/sub*
T0*-
_output_shapes
:�����������
�
Obert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/ShapeShapePbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/add_1*
T0*
_output_shapes
:
�
]bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
_bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
_bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Wbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/strided_sliceStridedSliceObert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/Shape]bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/strided_slice/stack_bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/strided_slice/stack_1_bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
�
qbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*a
_classW
USloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel*
valueB"8  �  *
dtype0*
_output_shapes
:
�
pbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*a
_classW
USloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
rbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*a
_classW
USloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
�
{bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalqbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel/Initializer/truncated_normal/shape*
T0*a
_classW
USloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel*
dtype0* 
_output_shapes
:
��	
�
obert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel/Initializer/truncated_normal/mulMul{bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalrbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel/Initializer/truncated_normal/stddev*
T0*a
_classW
USloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel* 
_output_shapes
:
��	
�
kbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel/Initializer/truncated_normalAddobert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel/Initializer/truncated_normal/mulpbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel/Initializer/truncated_normal/mean*
T0*a
_classW
USloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel* 
_output_shapes
:
��	
�
Nbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel
VariableV2*
shape:
��	*a
_classW
USloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel*
dtype0* 
_output_shapes
:
��	
�
Ubert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel/AssignAssignNbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernelkbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel/Initializer/truncated_normal*
T0*a
_classW
USloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel* 
_output_shapes
:
��	
�
Sbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel/readIdentityNbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel*
T0*a
_classW
USloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel* 
_output_shapes
:
��	
�
nbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias*
valueB:�	*
dtype0*
_output_shapes
:
�
dbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias/Initializer/zeros/ConstConst*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
�
^bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias/Initializer/zerosFillnbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias/Initializer/zeros/shape_as_tensordbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias/Initializer/zeros/Const*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias*
_output_shapes	
:�	
�
Lbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias
VariableV2*
shape:�	*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias*
dtype0*
_output_shapes	
:�	
�
Sbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias/AssignAssignLbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias^bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias/Initializer/zeros*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias*
_output_shapes	
:�	
�
Qbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias/readIdentityLbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias*
_output_shapes	
:�	
�
ebert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
`bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose	TransposePbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/add_1ebert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose/perm*
T0*-
_output_shapes
:�����������
�
gbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
bbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose_1	TransposeSbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel/readgbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose_1/perm*
T0* 
_output_shapes
:
��	
�
\bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/ShapeShape`bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose*
T0*
_output_shapes
:
�
jbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
lbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
lbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
dbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/strided_sliceStridedSlice\bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Shapejbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/strided_slice/stacklbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/strided_slice/stack_1lbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
�
\bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/mul/xConst*
value	B :*
dtype0*
_output_shapes
: 
�
Zbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/mulMul\bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/mul/xdbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/strided_slice*
T0*
_output_shapes
: 
�
^bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/mul_1/yConst*
value
B :�*
dtype0*
_output_shapes
: 
�
\bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/mul_1MulZbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/mul^bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/mul_1/y*
T0*
_output_shapes
: 
�
fbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
dbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape/shapePack\bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/mul_1fbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
^bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/ReshapeReshape`bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transposedbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape/shape*
T0*(
_output_shapes
:����������
�
]bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/MatMulMatMul^bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshapebbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose_1*
T0*(
_output_shapes
:����������	
�
hbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape_1/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
hbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape_1/shape/2Const*
value
B :�	*
dtype0*
_output_shapes
: 
�
fbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape_1/shapePackdbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/strided_slicehbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape_1/shape/1hbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape_1/shape/2*
T0*
N*
_output_shapes
:
�
`bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape_1Reshape]bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/MatMulfbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape_1/shape*
T0*-
_output_shapes
:�����������	
�
gbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose_2/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
bbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose_2	Transpose`bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape_1gbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose_2/perm*
T0*-
_output_shapes
:�����������	
�
Sbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/addAddbbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose_2Qbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias/read*
T0*-
_output_shapes
:�����������	
�
Obert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/Pow/yConst*
valueB
 *  @@*
dtype0*
_output_shapes
: 
�
Mbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/PowPowSbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/addObert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/Pow/y*
T0*-
_output_shapes
:�����������	
�
Obert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul/xConst*
valueB
 *'7=*
dtype0*
_output_shapes
: 
�
Mbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mulMulObert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul/xMbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/Pow*
T0*-
_output_shapes
:�����������	
�
Mbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/addAddSbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/addMbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul*
T0*-
_output_shapes
:�����������	
�
Qbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul_1/xConst*
valueB
 **BL?*
dtype0*
_output_shapes
: 
�
Obert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul_1MulQbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul_1/xMbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/add*
T0*-
_output_shapes
:�����������	
�
Nbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/TanhTanhObert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul_1*
T0*-
_output_shapes
:�����������	
�
Qbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/add_1/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Obert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/add_1AddQbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/add_1/xNbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/Tanh*
T0*-
_output_shapes
:�����������	
�
Qbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul_2/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
Obert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul_2MulQbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul_2/xObert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/add_1*
T0*-
_output_shapes
:�����������	
�
Obert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul_3MulSbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/addObert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul_2*
T0*-
_output_shapes
:�����������	
�
Vbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/ShapeShapeObert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul_3*
T0*
_output_shapes
:
�
dbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
fbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
fbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
^bert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/strided_sliceStridedSliceVbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/Shapedbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/strided_slice/stackfbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/strided_slice/stack_1fbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
�
xbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel/Initializer/truncated_normal/shapeConst*h
_class^
\Zloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel*
valueB"�  8  *
dtype0*
_output_shapes
:
�
wbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel/Initializer/truncated_normal/meanConst*h
_class^
\Zloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
ybert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel/Initializer/truncated_normal/stddevConst*h
_class^
\Zloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
�
�bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalxbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel/Initializer/truncated_normal/shape*
T0*h
_class^
\Zloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel*
dtype0* 
_output_shapes
:
�	�
�
vbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel/Initializer/truncated_normal/mulMul�bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalybert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*h
_class^
\Zloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel* 
_output_shapes
:
�	�
�
rbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel/Initializer/truncated_normalAddvbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel/Initializer/truncated_normal/mulwbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel/Initializer/truncated_normal/mean*
T0*h
_class^
\Zloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel* 
_output_shapes
:
�	�
�
Ubert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel
VariableV2*
shape:
�	�*h
_class^
\Zloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel*
dtype0* 
_output_shapes
:
�	�
�
\bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel/AssignAssignUbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernelrbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel/Initializer/truncated_normal*
T0*h
_class^
\Zloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel* 
_output_shapes
:
�	�
�
Zbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel/readIdentityUbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel*
T0*h
_class^
\Zloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel* 
_output_shapes
:
�	�
�
ebert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias/Initializer/zerosConst*f
_class\
ZXloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Sbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias
VariableV2*f
_class\
ZXloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias*
dtype0*
_output_shapes	
:�*
shape:�
�
Zbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias/AssignAssignSbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/biasebert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias/Initializer/zeros*
T0*f
_class\
ZXloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias*
_output_shapes	
:�
�
Xbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias/readIdentitySbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias*
T0*f
_class\
ZXloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias*
_output_shapes	
:�
�
lbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
gbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose	TransposeObert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul_3lbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose/perm*
T0*-
_output_shapes
:�����������	
�
nbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
ibert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose_1	TransposeZbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel/readnbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose_1/perm*
T0* 
_output_shapes
:
�	�
�
cbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/ShapeShapegbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose*
T0*
_output_shapes
:
�
qbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
sbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
sbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
kbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/strided_sliceStridedSlicecbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Shapeqbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/strided_slice/stacksbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/strided_slice/stack_1sbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
�
cbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/mul/xConst*
value	B :*
dtype0*
_output_shapes
: 
�
abert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/mulMulcbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/mul/xkbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/strided_slice*
T0*
_output_shapes
: 
�
ebert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/mul_1/yConst*
value
B :�*
dtype0*
_output_shapes
: 
�
cbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/mul_1Mulabert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/mulebert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/mul_1/y*
T0*
_output_shapes
: 
�
mbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape/shape/1Const*
value
B :�	*
dtype0*
_output_shapes
: 
�
kbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape/shapePackcbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/mul_1mbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
ebert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/ReshapeReshapegbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transposekbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape/shape*
T0*(
_output_shapes
:����������	
�
dbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/MatMulMatMulebert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshapeibert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose_1*
T0*(
_output_shapes
:����������
�
obert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape_1/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
obert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape_1/shape/2Const*
value
B :�*
dtype0*
_output_shapes
: 
�
mbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape_1/shapePackkbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/strided_sliceobert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape_1/shape/1obert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape_1/shape/2*
T0*
N*
_output_shapes
:
�
gbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape_1Reshapedbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/MatMulmbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape_1/shape*
T0*-
_output_shapes
:�����������
�
nbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose_2/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
ibert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose_2	Transposegbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape_1nbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose_2/perm*
T0*-
_output_shapes
:�����������
�
Zbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/addAddibert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose_2Xbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias/read*
T0*-
_output_shapes
:�����������
�
<bert/encoder/transformer/group_0/layer_0/inner_group_0/add_1AddZbert/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/addPbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/add_1*
T0*-
_output_shapes
:�����������
�
Qbert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta/Initializer/zerosConst*R
_classH
FDloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
?bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta
VariableV2*
shape:�*R
_classH
FDloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta*
dtype0*
_output_shapes	
:�
�
Fbert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta/AssignAssign?bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/betaQbert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta/Initializer/zeros*
T0*R
_classH
FDloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta*
_output_shapes	
:�
�
Dbert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta/readIdentity?bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta*
T0*R
_classH
FDloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta*
_output_shapes	
:�
�
Qbert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma/Initializer/onesConst*S
_classI
GEloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
@bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma
VariableV2*
shape:�*S
_classI
GEloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma*
dtype0*
_output_shapes	
:�
�
Gbert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma/AssignAssign@bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gammaQbert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma/Initializer/ones*
T0*S
_classI
GEloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma*
_output_shapes	
:�
�
Ebert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma/readIdentity@bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma*
T0*S
_classI
GEloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma*
_output_shapes	
:�
�
abert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
Obert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/meanMean<bert/encoder/transformer/group_0/layer_0/inner_group_0/add_1abert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/mean/reduction_indices*
T0*,
_output_shapes
:����������*
	keep_dims(
�
Wbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/StopGradientStopGradientObert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/mean*
T0*,
_output_shapes
:����������
�
\bert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/SquaredDifferenceSquaredDifference<bert/encoder/transformer/group_0/layer_0/inner_group_0/add_1Wbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/StopGradient*
T0*-
_output_shapes
:�����������
�
ebert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
Sbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/varianceMean\bert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/SquaredDifferenceebert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/variance/reduction_indices*
T0*,
_output_shapes
:����������*
	keep_dims(
�
Rbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/add/yConst*
valueB
 *̼�+*
dtype0*
_output_shapes
: 
�
Pbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/addAddSbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/varianceRbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/add/y*
T0*,
_output_shapes
:����������
�
Rbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/RsqrtRsqrtPbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/add*
T0*,
_output_shapes
:����������
�
Pbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/mulMulRbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/RsqrtEbert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma/read*
T0*-
_output_shapes
:�����������
�
Rbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/mul_1Mul<bert/encoder/transformer/group_0/layer_0/inner_group_0/add_1Pbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/mul*
T0*-
_output_shapes
:�����������
�
Rbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/mul_2MulObert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/meanPbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/mul*
T0*-
_output_shapes
:�����������
�
Pbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/subSubDbert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta/readRbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/mul_2*
T0*-
_output_shapes
:�����������
�
Rbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/add_1AddRbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/mul_1Pbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/sub*
T0*-
_output_shapes
:�����������
t
bert/pooler/strided_slice/stackConst*!
valueB"            *
dtype0*
_output_shapes
:
v
!bert/pooler/strided_slice/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
v
!bert/pooler/strided_slice/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
�
bert/pooler/strided_sliceStridedSliceRbert/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/add_1bert/pooler/strided_slice/stack!bert/pooler/strided_slice/stack_1!bert/pooler/strided_slice/stack_2*
Index0*
T0*

begin_mask*
end_mask*,
_output_shapes
:����������
�
bert/pooler/SqueezeSqueezebert/pooler/strided_slice*
squeeze_dims
*
T0*(
_output_shapes
:����������
�
;bert/pooler/dense/kernel/Initializer/truncated_normal/shapeConst*+
_class!
loc:@bert/pooler/dense/kernel*
valueB"8  8  *
dtype0*
_output_shapes
:
�
:bert/pooler/dense/kernel/Initializer/truncated_normal/meanConst*+
_class!
loc:@bert/pooler/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
<bert/pooler/dense/kernel/Initializer/truncated_normal/stddevConst*+
_class!
loc:@bert/pooler/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
�
Ebert/pooler/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal;bert/pooler/dense/kernel/Initializer/truncated_normal/shape*
T0*+
_class!
loc:@bert/pooler/dense/kernel*
dtype0* 
_output_shapes
:
��
�
9bert/pooler/dense/kernel/Initializer/truncated_normal/mulMulEbert/pooler/dense/kernel/Initializer/truncated_normal/TruncatedNormal<bert/pooler/dense/kernel/Initializer/truncated_normal/stddev*
T0*+
_class!
loc:@bert/pooler/dense/kernel* 
_output_shapes
:
��
�
5bert/pooler/dense/kernel/Initializer/truncated_normalAdd9bert/pooler/dense/kernel/Initializer/truncated_normal/mul:bert/pooler/dense/kernel/Initializer/truncated_normal/mean*
T0*+
_class!
loc:@bert/pooler/dense/kernel* 
_output_shapes
:
��
�
bert/pooler/dense/kernel
VariableV2*+
_class!
loc:@bert/pooler/dense/kernel*
dtype0* 
_output_shapes
:
��*
shape:
��
�
bert/pooler/dense/kernel/AssignAssignbert/pooler/dense/kernel5bert/pooler/dense/kernel/Initializer/truncated_normal*
T0*+
_class!
loc:@bert/pooler/dense/kernel* 
_output_shapes
:
��
�
bert/pooler/dense/kernel/readIdentitybert/pooler/dense/kernel*
T0*+
_class!
loc:@bert/pooler/dense/kernel* 
_output_shapes
:
��
�
(bert/pooler/dense/bias/Initializer/zerosConst*)
_class
loc:@bert/pooler/dense/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
bert/pooler/dense/bias
VariableV2*)
_class
loc:@bert/pooler/dense/bias*
dtype0*
_output_shapes	
:�*
shape:�
�
bert/pooler/dense/bias/AssignAssignbert/pooler/dense/bias(bert/pooler/dense/bias/Initializer/zeros*
T0*)
_class
loc:@bert/pooler/dense/bias*
_output_shapes	
:�
�
bert/pooler/dense/bias/readIdentitybert/pooler/dense/bias*
T0*)
_class
loc:@bert/pooler/dense/bias*
_output_shapes	
:�
�
bert/pooler/dense/MatMulMatMulbert/pooler/Squeezebert/pooler/dense/kernel/read*
T0*(
_output_shapes
:����������
�
bert/pooler/dense/BiasAddBiasAddbert/pooler/dense/MatMulbert/pooler/dense/bias/read*
T0*(
_output_shapes
:����������
l
bert/pooler/dense/TanhTanhbert/pooler/dense/BiasAdd*
T0*(
_output_shapes
:����������
B
Shape_2Shapeb_input_ids*
T0	*
_output_shapes
:
_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
strided_slice_1StridedSliceShape_2strided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
s
 bert_1/embeddings/ExpandDims/dimConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
bert_1/embeddings/ExpandDims
ExpandDimsb_input_ids bert_1/embeddings/ExpandDims/dim*
T0	*,
_output_shapes
:����������
�
'bert_1/embeddings/embedding_lookup/axisConst*2
_class(
&$loc:@bert/embeddings/word_embeddings*
value	B : *
dtype0*
_output_shapes
: 
�
"bert_1/embeddings/embedding_lookupGatherV2$bert/embeddings/word_embeddings/readbert_1/embeddings/ExpandDims'bert_1/embeddings/embedding_lookup/axis*
Taxis0*
Tindices0	*
Tparams0*2
_class(
&$loc:@bert/embeddings/word_embeddings*1
_output_shapes
:�����������
�
+bert_1/embeddings/embedding_lookup/IdentityIdentity"bert_1/embeddings/embedding_lookup*
T0*1
_output_shapes
:�����������
c
bert_1/embeddings/ShapeShapebert_1/embeddings/ExpandDims*
T0	*
_output_shapes
:
o
%bert_1/embeddings/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
q
'bert_1/embeddings/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
q
'bert_1/embeddings/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
bert_1/embeddings/strided_sliceStridedSlicebert_1/embeddings/Shape%bert_1/embeddings/strided_slice/stack'bert_1/embeddings/strided_slice/stack_1'bert_1/embeddings/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
d
!bert_1/embeddings/Reshape/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
d
!bert_1/embeddings/Reshape/shape/2Const*
value
B :�*
dtype0*
_output_shapes
: 
�
bert_1/embeddings/Reshape/shapePackbert_1/embeddings/strided_slice!bert_1/embeddings/Reshape/shape/1!bert_1/embeddings/Reshape/shape/2*
T0*
N*
_output_shapes
:
�
bert_1/embeddings/ReshapeReshape+bert_1/embeddings/embedding_lookup/Identitybert_1/embeddings/Reshape/shape*
T0*-
_output_shapes
:�����������
b
bert_1/embeddings/Shape_1Shapebert_1/embeddings/Reshape*
T0*
_output_shapes
:
q
'bert_1/embeddings/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
s
)bert_1/embeddings/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
s
)bert_1/embeddings/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
!bert_1/embeddings/strided_slice_1StridedSlicebert_1/embeddings/Shape_1'bert_1/embeddings/strided_slice_1/stack)bert_1/embeddings/strided_slice_1/stack_1)bert_1/embeddings/strided_slice_1/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
t
!bert_1/embeddings/Reshape_1/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
�
bert_1/embeddings/Reshape_1Reshapeb_segment_ids!bert_1/embeddings/Reshape_1/shape*
T0	*#
_output_shapes
:���������
g
"bert_1/embeddings/one_hot/on_valueConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
h
#bert_1/embeddings/one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
a
bert_1/embeddings/one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 
�
bert_1/embeddings/one_hotOneHotbert_1/embeddings/Reshape_1bert_1/embeddings/one_hot/depth"bert_1/embeddings/one_hot/on_value#bert_1/embeddings/one_hot/off_value*
T0*'
_output_shapes
:���������
�
bert_1/embeddings/MatMulMatMulbert_1/embeddings/one_hot*bert/embeddings/token_type_embeddings/read*
T0*(
_output_shapes
:����������
f
#bert_1/embeddings/Reshape_2/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
f
#bert_1/embeddings/Reshape_2/shape/2Const*
value
B :�*
dtype0*
_output_shapes
: 
�
!bert_1/embeddings/Reshape_2/shapePack!bert_1/embeddings/strided_slice_1#bert_1/embeddings/Reshape_2/shape/1#bert_1/embeddings/Reshape_2/shape/2*
T0*
N*
_output_shapes
:
�
bert_1/embeddings/Reshape_2Reshapebert_1/embeddings/MatMul!bert_1/embeddings/Reshape_2/shape*
T0*-
_output_shapes
:�����������
�
bert_1/embeddings/addAddbert_1/embeddings/Reshapebert_1/embeddings/Reshape_2*
T0*-
_output_shapes
:�����������
h
%bert_1/embeddings/assert_less_equal/xConst*
value
B :�*
dtype0*
_output_shapes
: 
h
%bert_1/embeddings/assert_less_equal/yConst*
value
B :�*
dtype0*
_output_shapes
: 
�
-bert_1/embeddings/assert_less_equal/LessEqual	LessEqual%bert_1/embeddings/assert_less_equal/x%bert_1/embeddings/assert_less_equal/y*
T0*
_output_shapes
: 
l
)bert_1/embeddings/assert_less_equal/ConstConst*
valueB *
dtype0*
_output_shapes
: 
�
'bert_1/embeddings/assert_less_equal/AllAll-bert_1/embeddings/assert_less_equal/LessEqual)bert_1/embeddings/assert_less_equal/Const*
_output_shapes
: 
q
0bert_1/embeddings/assert_less_equal/Assert/ConstConst*
valueB B *
dtype0*
_output_shapes
: 
�
2bert_1/embeddings/assert_less_equal/Assert/Const_1Const*j
valueaB_ BYCondition x <= y did not hold element-wise:x (bert_1/embeddings/assert_less_equal/x:0) = *
dtype0*
_output_shapes
: 
�
2bert_1/embeddings/assert_less_equal/Assert/Const_2Const*?
value6B4 B.y (bert_1/embeddings/assert_less_equal/y:0) = *
dtype0*
_output_shapes
: 
y
8bert_1/embeddings/assert_less_equal/Assert/Assert/data_0Const*
valueB B *
dtype0*
_output_shapes
: 
�
8bert_1/embeddings/assert_less_equal/Assert/Assert/data_1Const*j
valueaB_ BYCondition x <= y did not hold element-wise:x (bert_1/embeddings/assert_less_equal/x:0) = *
dtype0*
_output_shapes
: 
�
8bert_1/embeddings/assert_less_equal/Assert/Assert/data_3Const*?
value6B4 B.y (bert_1/embeddings/assert_less_equal/y:0) = *
dtype0*
_output_shapes
: 
�
1bert_1/embeddings/assert_less_equal/Assert/AssertAssert'bert_1/embeddings/assert_less_equal/All8bert_1/embeddings/assert_less_equal/Assert/Assert/data_08bert_1/embeddings/assert_less_equal/Assert/Assert/data_1%bert_1/embeddings/assert_less_equal/x8bert_1/embeddings/assert_less_equal/Assert/Assert/data_3%bert_1/embeddings/assert_less_equal/y*
T	
2
�
bert_1/embeddings/Slice/beginConst2^bert_1/embeddings/assert_less_equal/Assert/Assert*
valueB"        *
dtype0*
_output_shapes
:
�
bert_1/embeddings/Slice/sizeConst2^bert_1/embeddings/assert_less_equal/Assert/Assert*
valueB"�   ����*
dtype0*
_output_shapes
:
�
bert_1/embeddings/SliceSlice(bert/embeddings/position_embeddings/readbert_1/embeddings/Slice/beginbert_1/embeddings/Slice/size*
Index0*
T0* 
_output_shapes
:
��
�
!bert_1/embeddings/Reshape_3/shapeConst2^bert_1/embeddings/assert_less_equal/Assert/Assert*!
valueB"   �   �   *
dtype0*
_output_shapes
:
�
bert_1/embeddings/Reshape_3Reshapebert_1/embeddings/Slice!bert_1/embeddings/Reshape_3/shape*
T0*$
_output_shapes
:��
�
bert_1/embeddings/add_1Addbert_1/embeddings/addbert_1/embeddings/Reshape_3*
T0*-
_output_shapes
:�����������
�
:bert_1/embeddings/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
(bert_1/embeddings/LayerNorm/moments/meanMeanbert_1/embeddings/add_1:bert_1/embeddings/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*
T0*,
_output_shapes
:����������
�
0bert_1/embeddings/LayerNorm/moments/StopGradientStopGradient(bert_1/embeddings/LayerNorm/moments/mean*
T0*,
_output_shapes
:����������
�
5bert_1/embeddings/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert_1/embeddings/add_10bert_1/embeddings/LayerNorm/moments/StopGradient*
T0*-
_output_shapes
:�����������
�
>bert_1/embeddings/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
,bert_1/embeddings/LayerNorm/moments/varianceMean5bert_1/embeddings/LayerNorm/moments/SquaredDifference>bert_1/embeddings/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*
T0*,
_output_shapes
:����������
p
+bert_1/embeddings/LayerNorm/batchnorm/add/yConst*
valueB
 *̼�+*
dtype0*
_output_shapes
: 
�
)bert_1/embeddings/LayerNorm/batchnorm/addAdd,bert_1/embeddings/LayerNorm/moments/variance+bert_1/embeddings/LayerNorm/batchnorm/add/y*
T0*,
_output_shapes
:����������
�
+bert_1/embeddings/LayerNorm/batchnorm/RsqrtRsqrt)bert_1/embeddings/LayerNorm/batchnorm/add*
T0*,
_output_shapes
:����������
�
)bert_1/embeddings/LayerNorm/batchnorm/mulMul+bert_1/embeddings/LayerNorm/batchnorm/Rsqrt$bert/embeddings/LayerNorm/gamma/read*
T0*-
_output_shapes
:�����������
�
+bert_1/embeddings/LayerNorm/batchnorm/mul_1Mulbert_1/embeddings/add_1)bert_1/embeddings/LayerNorm/batchnorm/mul*
T0*-
_output_shapes
:�����������
�
+bert_1/embeddings/LayerNorm/batchnorm/mul_2Mul(bert_1/embeddings/LayerNorm/moments/mean)bert_1/embeddings/LayerNorm/batchnorm/mul*
T0*-
_output_shapes
:�����������
�
)bert_1/embeddings/LayerNorm/batchnorm/subSub#bert/embeddings/LayerNorm/beta/read+bert_1/embeddings/LayerNorm/batchnorm/mul_2*
T0*-
_output_shapes
:�����������
�
+bert_1/embeddings/LayerNorm/batchnorm/add_1Add+bert_1/embeddings/LayerNorm/batchnorm/mul_1)bert_1/embeddings/LayerNorm/batchnorm/sub*
T0*-
_output_shapes
:�����������
o
bert_1/encoder/ShapeShape+bert_1/embeddings/LayerNorm/batchnorm/add_1*
T0*
_output_shapes
:
l
"bert_1/encoder/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
n
$bert_1/encoder/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
n
$bert_1/encoder/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
bert_1/encoder/strided_sliceStridedSlicebert_1/encoder/Shape"bert_1/encoder/strided_slice/stack$bert_1/encoder/strided_slice/stack_1$bert_1/encoder/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
q
bert_1/encoder/Shape_1Shape+bert_1/embeddings/LayerNorm/batchnorm/add_1*
T0*
_output_shapes
:
n
$bert_1/encoder/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
p
&bert_1/encoder/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
p
&bert_1/encoder/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
bert_1/encoder/strided_slice_1StridedSlicebert_1/encoder/Shape_1$bert_1/encoder/strided_slice_1/stack&bert_1/encoder/strided_slice_1/stack_1&bert_1/encoder/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
�
@bert_1/encoder/embedding_hidden_mapping_in/einsum/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
;bert_1/encoder/embedding_hidden_mapping_in/einsum/transpose	Transpose+bert_1/embeddings/LayerNorm/batchnorm/add_1@bert_1/encoder/embedding_hidden_mapping_in/einsum/transpose/perm*
T0*-
_output_shapes
:�����������
�
Bbert_1/encoder/embedding_hidden_mapping_in/einsum/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
=bert_1/encoder/embedding_hidden_mapping_in/einsum/transpose_1	Transpose4bert/encoder/embedding_hidden_mapping_in/kernel/readBbert_1/encoder/embedding_hidden_mapping_in/einsum/transpose_1/perm*
T0* 
_output_shapes
:
��
�
7bert_1/encoder/embedding_hidden_mapping_in/einsum/ShapeShape;bert_1/encoder/embedding_hidden_mapping_in/einsum/transpose*
T0*
_output_shapes
:
�
Ebert_1/encoder/embedding_hidden_mapping_in/einsum/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
Gbert_1/encoder/embedding_hidden_mapping_in/einsum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
Gbert_1/encoder/embedding_hidden_mapping_in/einsum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
?bert_1/encoder/embedding_hidden_mapping_in/einsum/strided_sliceStridedSlice7bert_1/encoder/embedding_hidden_mapping_in/einsum/ShapeEbert_1/encoder/embedding_hidden_mapping_in/einsum/strided_slice/stackGbert_1/encoder/embedding_hidden_mapping_in/einsum/strided_slice/stack_1Gbert_1/encoder/embedding_hidden_mapping_in/einsum/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
y
7bert_1/encoder/embedding_hidden_mapping_in/einsum/mul/xConst*
value	B :*
dtype0*
_output_shapes
: 
�
5bert_1/encoder/embedding_hidden_mapping_in/einsum/mulMul7bert_1/encoder/embedding_hidden_mapping_in/einsum/mul/x?bert_1/encoder/embedding_hidden_mapping_in/einsum/strided_slice*
T0*
_output_shapes
: 
|
9bert_1/encoder/embedding_hidden_mapping_in/einsum/mul_1/yConst*
value
B :�*
dtype0*
_output_shapes
: 
�
7bert_1/encoder/embedding_hidden_mapping_in/einsum/mul_1Mul5bert_1/encoder/embedding_hidden_mapping_in/einsum/mul9bert_1/encoder/embedding_hidden_mapping_in/einsum/mul_1/y*
T0*
_output_shapes
: 
�
Abert_1/encoder/embedding_hidden_mapping_in/einsum/Reshape/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
?bert_1/encoder/embedding_hidden_mapping_in/einsum/Reshape/shapePack7bert_1/encoder/embedding_hidden_mapping_in/einsum/mul_1Abert_1/encoder/embedding_hidden_mapping_in/einsum/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
9bert_1/encoder/embedding_hidden_mapping_in/einsum/ReshapeReshape;bert_1/encoder/embedding_hidden_mapping_in/einsum/transpose?bert_1/encoder/embedding_hidden_mapping_in/einsum/Reshape/shape*
T0*(
_output_shapes
:����������
�
8bert_1/encoder/embedding_hidden_mapping_in/einsum/MatMulMatMul9bert_1/encoder/embedding_hidden_mapping_in/einsum/Reshape=bert_1/encoder/embedding_hidden_mapping_in/einsum/transpose_1*
T0*(
_output_shapes
:����������
�
Cbert_1/encoder/embedding_hidden_mapping_in/einsum/Reshape_1/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
Cbert_1/encoder/embedding_hidden_mapping_in/einsum/Reshape_1/shape/2Const*
value
B :�*
dtype0*
_output_shapes
: 
�
Abert_1/encoder/embedding_hidden_mapping_in/einsum/Reshape_1/shapePack?bert_1/encoder/embedding_hidden_mapping_in/einsum/strided_sliceCbert_1/encoder/embedding_hidden_mapping_in/einsum/Reshape_1/shape/1Cbert_1/encoder/embedding_hidden_mapping_in/einsum/Reshape_1/shape/2*
T0*
N*
_output_shapes
:
�
;bert_1/encoder/embedding_hidden_mapping_in/einsum/Reshape_1Reshape8bert_1/encoder/embedding_hidden_mapping_in/einsum/MatMulAbert_1/encoder/embedding_hidden_mapping_in/einsum/Reshape_1/shape*
T0*-
_output_shapes
:�����������
�
Bbert_1/encoder/embedding_hidden_mapping_in/einsum/transpose_2/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
=bert_1/encoder/embedding_hidden_mapping_in/einsum/transpose_2	Transpose;bert_1/encoder/embedding_hidden_mapping_in/einsum/Reshape_1Bbert_1/encoder/embedding_hidden_mapping_in/einsum/transpose_2/perm*
T0*-
_output_shapes
:�����������
�
.bert_1/encoder/embedding_hidden_mapping_in/addAdd=bert_1/encoder/embedding_hidden_mapping_in/einsum/transpose_22bert/encoder/embedding_hidden_mapping_in/bias/read*
T0*-
_output_shapes
:�����������
�
Obert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ShapeShape.bert_1/encoder/embedding_hidden_mapping_in/add*
T0*
_output_shapes
:
�
]bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Wbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_sliceStridedSliceObert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape]bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice/stack_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice/stack_1_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
�
Qbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_1Shape.bert_1/encoder/embedding_hidden_mapping_in/add*
T0*
_output_shapes
:
�
_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
abert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
abert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Ybert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_1StridedSliceQbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_1_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_1/stackabert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_1/stack_1abert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_1/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
�
Qbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_2Shape.bert_1/encoder/embedding_hidden_mapping_in/add*
T0*
_output_shapes
:
�
_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
abert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
abert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Ybert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_2StridedSliceQbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_2_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_2/stackabert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_2/stack_1abert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_2/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
�
]bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/Reshape/shapeConst*!
valueB"8        *
dtype0*
_output_shapes
:
�
Wbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/ReshapeReshapeQbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel/read]bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/Reshape/shape*
T0*#
_output_shapes
:�
�
_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/Reshape_1/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
Ybert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/Reshape_1ReshapeObert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias/read_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/Reshape_1/shape*
T0*
_output_shapes

:
�
ebert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose	Transpose.bert_1/encoder/embedding_hidden_mapping_in/addebert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose/perm*
T0*-
_output_shapes
:�����������
�
gbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
bbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose_1	TransposeWbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/Reshapegbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose_1/perm*
T0*#
_output_shapes
:�
�
\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/ShapeShape`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose*
T0*
_output_shapes
:
�
jbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
lbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
lbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
dbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/strided_sliceStridedSlice\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Shapejbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/strided_slice/stacklbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/strided_slice/stack_1lbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
�
\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/mul/xConst*
value	B :*
dtype0*
_output_shapes
: 
�
Zbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/mulMul\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/mul/xdbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/strided_slice*
T0*
_output_shapes
: 
�
^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/mul_1/yConst*
dtype0*
_output_shapes
: *
value
B :�
�
\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/mul_1MulZbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/mul^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/mul_1/y*
T0*
_output_shapes
: 
�
fbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value
B :�
�
dbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape/shapePack\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/mul_1fbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/ReshapeReshape`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transposedbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape/shape*
T0*(
_output_shapes
:����������
�
fbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_1/shapeConst*
valueB"8  8  *
dtype0*
_output_shapes
:
�
`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_1Reshapebbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose_1fbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_1/shape*
T0* 
_output_shapes
:
��
�
]bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/MatMulMatMul^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_1*
T0*(
_output_shapes
:����������
�
hbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_2/shape/1Const*
dtype0*
_output_shapes
: *
value
B :�
�
hbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_2/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
�
hbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_2/shape/3Const*
_output_shapes
: *
value	B :*
dtype0
�
fbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_2/shapePackdbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/strided_slicehbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_2/shape/1hbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_2/shape/2hbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_2/shape/3*
T0*
N*
_output_shapes
:
�
`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_2Reshape]bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/MatMulfbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_2/shape*0
_output_shapes
:����������*
T0
�
gbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose_2/permConst*
dtype0*
_output_shapes
:*%
valueB"             
�
bbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose_2	Transpose`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/Reshape_2gbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose_2/perm*
T0*0
_output_shapes
:����������
�
Sbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/addAddbbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum/transpose_2Ybert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/Reshape_1*0
_output_shapes
:����������*
T0
�
Qbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_3Shape.bert_1/encoder/embedding_hidden_mapping_in/add*
T0*
_output_shapes
:
�
_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
abert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
abert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
Ybert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_3StridedSliceQbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_3_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_3/stackabert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_3/stack_1abert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_3/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
�
[bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/Reshape/shapeConst*!
valueB"8        *
dtype0*
_output_shapes
:
�
Ubert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/ReshapeReshapeObert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel/read[bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/Reshape/shape*
T0*#
_output_shapes
:�
�
]bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/Reshape_1/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
Wbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/Reshape_1ReshapeMbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias/read]bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/Reshape_1/shape*
_output_shapes

:*
T0
�
cbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose	Transpose.bert_1/encoder/embedding_hidden_mapping_in/addcbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose/perm*
T0*-
_output_shapes
:�����������
�
ebert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose_1/permConst*
dtype0*
_output_shapes
:*!
valueB"          
�
`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose_1	TransposeUbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/Reshapeebert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose_1/perm*
T0*#
_output_shapes
:�
�
Zbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/ShapeShape^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose*
T0*
_output_shapes
:
�
hbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
jbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
jbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
bbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/strided_sliceStridedSliceZbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Shapehbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/strided_slice/stackjbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/strided_slice/stack_1jbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
�
Zbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/mul/xConst*
dtype0*
_output_shapes
: *
value	B :
�
Xbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/mulMulZbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/mul/xbbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/strided_slice*
T0*
_output_shapes
: 
�
\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/mul_1/yConst*
value
B :�*
dtype0*
_output_shapes
: 
�
Zbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/mul_1MulXbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/mul\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/mul_1/y*
_output_shapes
: *
T0
�
dbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
bbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape/shapePackZbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/mul_1dbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/ReshapeReshape^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transposebbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape/shape*(
_output_shapes
:����������*
T0
�
dbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_1/shapeConst*
valueB"8  8  *
dtype0*
_output_shapes
:
�
^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_1Reshape`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose_1dbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_1/shape*
T0* 
_output_shapes
:
��
�
[bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/MatMulMatMul\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_1*
T0*(
_output_shapes
:����������
�
fbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_2/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
fbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_2/shape/2Const*
dtype0*
_output_shapes
: *
value	B :
�
fbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_2/shape/3Const*
value	B :*
dtype0*
_output_shapes
: 
�
dbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_2/shapePackbbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/strided_slicefbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_2/shape/1fbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_2/shape/2fbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_2/shape/3*
N*
_output_shapes
:*
T0
�
^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_2Reshape[bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/MatMuldbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_2/shape*
T0*0
_output_shapes
:����������
�
ebert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
�
`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose_2	Transpose^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/Reshape_2ebert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose_2/perm*
T0*0
_output_shapes
:����������
�
Qbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/addAdd`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/einsum/transpose_2Wbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/Reshape_1*
T0*0
_output_shapes
:����������
�
Qbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_4Shape.bert_1/encoder/embedding_hidden_mapping_in/add*
T0*
_output_shapes
:
�
_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_4/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
abert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_4/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
abert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Ybert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_4StridedSliceQbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_4_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_4/stackabert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_4/stack_1abert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_4/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
�
]bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/Reshape/shapeConst*!
valueB"8        *
dtype0*
_output_shapes
:
�
Wbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/ReshapeReshapeQbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel/read]bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/Reshape/shape*#
_output_shapes
:�*
T0
�
_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/Reshape_1/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
�
Ybert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/Reshape_1ReshapeObert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias/read_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/Reshape_1/shape*
_output_shapes

:*
T0
�
ebert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose/permConst*
dtype0*
_output_shapes
:*!
valueB"          
�
`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose	Transpose.bert_1/encoder/embedding_hidden_mapping_in/addebert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose/perm*
T0*-
_output_shapes
:�����������
�
gbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
bbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose_1	TransposeWbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/Reshapegbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose_1/perm*
T0*#
_output_shapes
:�
�
\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/ShapeShape`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose*
T0*
_output_shapes
:
�
jbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
lbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
lbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
dbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/strided_sliceStridedSlice\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Shapejbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/strided_slice/stacklbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/strided_slice/stack_1lbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
�
\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/mul/xConst*
dtype0*
_output_shapes
: *
value	B :
�
Zbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/mulMul\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/mul/xdbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/strided_slice*
_output_shapes
: *
T0
�
^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/mul_1/yConst*
_output_shapes
: *
value
B :�*
dtype0
�
\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/mul_1MulZbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/mul^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/mul_1/y*
T0*
_output_shapes
: 
�
fbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
dbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape/shapePack\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/mul_1fbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape/shape/1*
T0*
N*
_output_shapes
:
�
^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/ReshapeReshape`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transposedbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape/shape*
T0*(
_output_shapes
:����������
�
fbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_1/shapeConst*
valueB"8  8  *
dtype0*
_output_shapes
:
�
`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_1Reshapebbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose_1fbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_1/shape*
T0* 
_output_shapes
:
��
�
]bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/MatMulMatMul^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_1*
T0*(
_output_shapes
:����������
�
hbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_2/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
hbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_2/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
�
hbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_2/shape/3Const*
value	B :*
dtype0*
_output_shapes
: 
�
fbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_2/shapePackdbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/strided_slicehbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_2/shape/1hbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_2/shape/2hbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_2/shape/3*
T0*
N*
_output_shapes
:
�
`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_2Reshape]bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/MatMulfbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_2/shape*
T0*0
_output_shapes
:����������
�
gbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose_2/permConst*
_output_shapes
:*%
valueB"             *
dtype0
�
bbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose_2	Transpose`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/Reshape_2gbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose_2/perm*0
_output_shapes
:����������*
T0
�
Sbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/addAddbbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/einsum/transpose_2Ybert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/Reshape_1*0
_output_shapes
:����������*
T0
�
Xbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
�
Sbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose	TransposeSbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/addXbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose/perm*
T0*0
_output_shapes
:����������
�
Zbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
�
Ubert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_1	TransposeQbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/key/addZbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_1/perm*0
_output_shapes
:����������*
T0
�
Zbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_2/permConst*
_output_shapes
:*%
valueB"             *
dtype0
�
Ubert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_2	TransposeSbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/value/addZbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_2/perm*
T0*0
_output_shapes
:����������
�
Ybert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Ybert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Reshape/shape/2Const*
value
B :�*
dtype0*
_output_shapes
: 
�
Ybert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Reshape/shape/3Const*
_output_shapes
: *
value	B :*
dtype0
�
Wbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Reshape/shapePackWbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_sliceYbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Reshape/shape/1Ybert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Reshape/shape/2Ybert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Reshape/shape/3*
T0*
N*
_output_shapes
:
�
Qbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ReshapeReshapeb_input_maskWbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Reshape/shape*
T0	*0
_output_shapes
:����������
�
Pbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/MatMulBatchMatMulSbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transposeUbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_1*1
_output_shapes
:�����������*
adj_y(*
T0
�
Qbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_5ShapeSbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose*
T0*
_output_shapes
:
�
_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_5/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
abert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_5/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
abert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_5/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Ybert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_5StridedSliceQbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_5_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_5/stackabert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_5/stack_1abert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_5/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
�
Obert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Mul/yConst*
valueB
 *��H>*
dtype0*
_output_shapes
: 
�
Mbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/MulMulPbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/MatMulObert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Mul/y*1
_output_shapes
:�����������*
T0
�
Qbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_6ShapeSbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose*
T0*
_output_shapes
:
�
_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_6/stackConst*
dtype0*
_output_shapes
:*
valueB: 
�
abert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_6/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
abert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_6/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Ybert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_6StridedSliceQbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Shape_6_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_6/stackabert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_6/stack_1abert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_6/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
�
Tbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mul/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
Rbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mulMulYbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_6Tbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mul/y*
_output_shapes
: *
T0
�
Vbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mul_1/yConst*
value
B :�*
dtype0*
_output_shapes
: 
�
Tbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mul_1MulRbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mulVbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mul_1/y*
T0*
_output_shapes
: 
�
Vbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mul_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
Tbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mul_2MulTbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mul_1Vbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mul_2/y*
_output_shapes
: *
T0
�
Ubert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/Less/yConst*
dtype0*
_output_shapes
: *
value
B :�
�
Sbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/LessLessTbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/mul_2Ubert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/Less/y*
_output_shapes
: *
T0
�
Wbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/packed/1Const*
value	B :*
dtype0*
_output_shapes
: 
�
Wbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/packed/2Const*
value
B :�*
dtype0*
_output_shapes
: 
�
Wbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/packed/3Const*
value	B :*
dtype0*
_output_shapes
: 
�
Ubert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/packedPackYbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/strided_slice_6Wbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/packed/1Wbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/packed/2Wbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/packed/3*
_output_shapes
:*
T0*
N
�
Tbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/onesFillUbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/packedTbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/ones/Const*
T0*0
_output_shapes
:����������
�
Nbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/CastCastQbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Reshape*

SrcT0	*0
_output_shapes
:����������*

DstT0
�
Rbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/MatMul_1BatchMatMulNbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/onesNbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/Cast*1
_output_shapes
:�����������*
adj_y(*
T0
�
Obert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Mbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/subSubObert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/sub/xRbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/MatMul_1*1
_output_shapes
:�����������*
T0
�
Qbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/mul_1/yConst*
valueB
 * @�*
dtype0*
_output_shapes
: 
�
Obert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/mul_1MulMbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/subQbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/mul_1/y*1
_output_shapes
:�����������*
T0
�
Mbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/addAddMbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/MulObert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/mul_1*
T0*1
_output_shapes
:�����������
�
Ybert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/attention_probsSoftmaxMbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/add*
T0*1
_output_shapes
:�����������
�
Rbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/MatMul_2BatchMatMulYbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/attention_probsUbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_2*
T0*0
_output_shapes
:����������
�
Zbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_3/permConst*
_output_shapes
:*%
valueB"             *
dtype0
�
Ubert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_3	TransposeRbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/MatMul_2Zbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_3/perm*
T0*0
_output_shapes
:����������
�
Qbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/ShapeShapeUbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_3*
_output_shapes
:*
T0
�
_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
�
abert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
abert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Ybert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/strided_sliceStridedSliceQbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/Shape_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/strided_slice/stackabert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/strided_slice/stack_1abert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
�
_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/Reshape/shapeConst*!
valueB"      8  *
dtype0*
_output_shapes
:
�
Ybert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/ReshapeReshapeSbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel/read_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/Reshape/shape*
T0*#
_output_shapes
:�
�
gbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose/permConst*
_output_shapes
:*%
valueB"             *
dtype0
�
bbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose	TransposeUbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/transpose_3gbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose/perm*
T0*0
_output_shapes
:����������
�
ibert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose_1/permConst*
_output_shapes
:*!
valueB"          *
dtype0
�
dbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose_1	TransposeYbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/Reshapeibert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose_1/perm*
T0*#
_output_shapes
:�
�
^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/ShapeShapebbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose*
T0*
_output_shapes
:
�
lbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
nbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
nbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
fbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/strided_sliceStridedSlice^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Shapelbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/strided_slice/stacknbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/strided_slice/stack_1nbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
�
^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/mul/xConst*
value	B :*
dtype0*
_output_shapes
: 
�
\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/mulMul^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/mul/xfbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/strided_slice*
T0*
_output_shapes
: 
�
`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/mul_1/yConst*
value
B :�*
dtype0*
_output_shapes
: 
�
^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/mul_1Mul\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/mul`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/mul_1/y*
T0*
_output_shapes
: 
�
hbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
fbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape/shapePack^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/mul_1hbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape/shape/1*
N*
_output_shapes
:*
T0
�
`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/ReshapeReshapebbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transposefbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape/shape*
T0*(
_output_shapes
:����������
�
hbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_1/shapeConst*
valueB"8  8  *
dtype0*
_output_shapes
:
�
bbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_1Reshapedbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose_1hbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_1/shape* 
_output_shapes
:
��*
T0
�
_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/MatMulMatMul`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshapebbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_1*
T0*(
_output_shapes
:����������
�
jbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_2/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
jbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_2/shape/2Const*
value
B :�*
dtype0*
_output_shapes
: 
�
hbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_2/shapePackfbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/strided_slicejbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_2/shape/1jbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_2/shape/2*
T0*
N*
_output_shapes
:
�
bbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_2Reshape_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/MatMulhbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_2/shape*
T0*-
_output_shapes
:�����������
�
ibert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose_2/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
dbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose_2	Transposebbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/Reshape_2ibert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose_2/perm*
T0*-
_output_shapes
:�����������
�
Ubert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/addAdddbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/einsum/transpose_2Qbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias/read*-
_output_shapes
:�����������*
T0
�
<bert_1/encoder/transformer/group_0/layer_0/inner_group_0/addAddUbert_1/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/output/dense/add.bert_1/encoder/embedding_hidden_mapping_in/add*-
_output_shapes
:�����������*
T0
�
abert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
�
Obert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/meanMean<bert_1/encoder/transformer/group_0/layer_0/inner_group_0/addabert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/mean/reduction_indices*
T0*,
_output_shapes
:����������*
	keep_dims(
�
Wbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/StopGradientStopGradientObert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/mean*
T0*,
_output_shapes
:����������
�
\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/SquaredDifferenceSquaredDifference<bert_1/encoder/transformer/group_0/layer_0/inner_group_0/addWbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/StopGradient*
T0*-
_output_shapes
:�����������
�
ebert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
Sbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/varianceMean\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/SquaredDifferenceebert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/variance/reduction_indices*,
_output_shapes
:����������*
	keep_dims(*
T0
�
Rbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/add/yConst*
valueB
 *̼�+*
dtype0*
_output_shapes
: 
�
Pbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/addAddSbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/varianceRbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/add/y*,
_output_shapes
:����������*
T0
�
Rbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/RsqrtRsqrtPbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/add*,
_output_shapes
:����������*
T0
�
Pbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/mulMulRbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/RsqrtCbert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma/read*
T0*-
_output_shapes
:�����������
�
Rbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/mul_1Mul<bert_1/encoder/transformer/group_0/layer_0/inner_group_0/addPbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/mul*-
_output_shapes
:�����������*
T0
�
Rbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/mul_2MulObert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/moments/meanPbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/mul*-
_output_shapes
:�����������*
T0
�
Pbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/subSubBbert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta/readRbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/mul_2*-
_output_shapes
:�����������*
T0
�
Rbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/add_1AddRbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/mul_1Pbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/sub*-
_output_shapes
:�����������*
T0
�
Qbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/ShapeShapeRbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/add_1*
T0*
_output_shapes
:
�
_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
abert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
abert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
Ybert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/strided_sliceStridedSliceQbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/Shape_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/strided_slice/stackabert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/strided_slice/stack_1abert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
�
gbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose/permConst*
dtype0*
_output_shapes
:*!
valueB"          
�
bbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose	TransposeRbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/add_1gbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose/perm*
T0*-
_output_shapes
:�����������
�
ibert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
dbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose_1	TransposeSbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel/readibert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose_1/perm*
T0* 
_output_shapes
:
��	
�
^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/ShapeShapebbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose*
_output_shapes
:*
T0
�
lbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
nbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
nbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
fbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/strided_sliceStridedSlice^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Shapelbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/strided_slice/stacknbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/strided_slice/stack_1nbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
�
^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/mul/xConst*
value	B :*
dtype0*
_output_shapes
: 
�
\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/mulMul^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/mul/xfbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/strided_slice*
_output_shapes
: *
T0
�
`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/mul_1/yConst*
dtype0*
_output_shapes
: *
value
B :�
�
^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/mul_1Mul\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/mul`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/mul_1/y*
T0*
_output_shapes
: 
�
hbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
fbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape/shapePack^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/mul_1hbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape/shape/1*
_output_shapes
:*
T0*
N
�
`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/ReshapeReshapebbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transposefbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape/shape*
T0*(
_output_shapes
:����������
�
_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/MatMulMatMul`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshapedbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose_1*
T0*(
_output_shapes
:����������	
�
jbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape_1/shape/1Const*
value
B :�*
dtype0*
_output_shapes
: 
�
jbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape_1/shape/2Const*
value
B :�	*
dtype0*
_output_shapes
: 
�
hbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape_1/shapePackfbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/strided_slicejbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape_1/shape/1jbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape_1/shape/2*
N*
_output_shapes
:*
T0
�
bbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape_1Reshape_bert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/MatMulhbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape_1/shape*
T0*-
_output_shapes
:�����������	
�
ibert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose_2/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
dbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose_2	Transposebbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/Reshape_1ibert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose_2/perm*
T0*-
_output_shapes
:�����������	
�
Ubert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/addAdddbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/einsum/transpose_2Qbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias/read*-
_output_shapes
:�����������	*
T0
�
Qbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/Pow/yConst*
dtype0*
_output_shapes
: *
valueB
 *  @@
�
Obert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/PowPowUbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/addQbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/Pow/y*
T0*-
_output_shapes
:�����������	
�
Qbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul/xConst*
valueB
 *'7=*
dtype0*
_output_shapes
: 
�
Obert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mulMulQbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul/xObert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/Pow*
T0*-
_output_shapes
:�����������	
�
Obert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/addAddUbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/addObert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul*
T0*-
_output_shapes
:�����������	
�
Sbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul_1/xConst*
valueB
 **BL?*
dtype0*
_output_shapes
: 
�
Qbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul_1MulSbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul_1/xObert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/add*
T0*-
_output_shapes
:�����������	
�
Pbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/TanhTanhQbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul_1*
T0*-
_output_shapes
:�����������	
�
Sbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/add_1/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Qbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/add_1AddSbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/add_1/xPbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/Tanh*
T0*-
_output_shapes
:�����������	
�
Sbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul_2/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
Qbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul_2MulSbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul_2/xQbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/add_1*
T0*-
_output_shapes
:�����������	
�
Qbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul_3MulUbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/dense/addQbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul_2*
T0*-
_output_shapes
:�����������	
�
Xbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/ShapeShapeQbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul_3*
T0*
_output_shapes
:
�
fbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
hbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
�
hbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
`bert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/strided_sliceStridedSliceXbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/Shapefbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/strided_slice/stackhbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/strided_slice/stack_1hbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
�
nbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
ibert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose	TransposeQbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/mul_3nbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose/perm*
T0*-
_output_shapes
:�����������	
�
pbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
kbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose_1	TransposeZbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel/readpbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose_1/perm* 
_output_shapes
:
�	�*
T0
�
ebert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/ShapeShapeibert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose*
T0*
_output_shapes
:
�
sbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
ubert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
�
ubert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
mbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/strided_sliceStridedSliceebert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Shapesbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/strided_slice/stackubert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/strided_slice/stack_1ubert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
�
ebert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/mul/xConst*
value	B :*
dtype0*
_output_shapes
: 
�
cbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/mulMulebert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/mul/xmbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/strided_slice*
_output_shapes
: *
T0
�
gbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/mul_1/yConst*
value
B :�*
dtype0*
_output_shapes
: 
�
ebert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/mul_1Mulcbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/mulgbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/mul_1/y*
T0*
_output_shapes
: 
�
obert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape/shape/1Const*
value
B :�	*
dtype0*
_output_shapes
: 
�
mbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape/shapePackebert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/mul_1obert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape/shape/1*
_output_shapes
:*
T0*
N
�
gbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/ReshapeReshapeibert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transposembert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape/shape*
T0*(
_output_shapes
:����������	
�
fbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/MatMulMatMulgbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshapekbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose_1*
T0*(
_output_shapes
:����������
�
qbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape_1/shape/1Const*
dtype0*
_output_shapes
: *
value
B :�
�
qbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape_1/shape/2Const*
dtype0*
_output_shapes
: *
value
B :�
�
obert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape_1/shapePackmbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/strided_sliceqbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape_1/shape/1qbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape_1/shape/2*
T0*
N*
_output_shapes
:
�
ibert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape_1Reshapefbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/MatMulobert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape_1/shape*
T0*-
_output_shapes
:�����������
�
pbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose_2/permConst*!
valueB"          *
dtype0*
_output_shapes
:
�
kbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose_2	Transposeibert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/Reshape_1pbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose_2/perm*-
_output_shapes
:�����������*
T0
�
\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/addAddkbert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/einsum/transpose_2Xbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias/read*
T0*-
_output_shapes
:�����������
�
>bert_1/encoder/transformer/group_0/layer_0/inner_group_0/add_1Add\bert_1/encoder/transformer/group_0/layer_0/inner_group_0/ffn_1/intermediate/output/dense/addRbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm/batchnorm/add_1*
T0*-
_output_shapes
:�����������
�
cbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
�
Qbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/meanMean>bert_1/encoder/transformer/group_0/layer_0/inner_group_0/add_1cbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/mean/reduction_indices*,
_output_shapes
:����������*
	keep_dims(*
T0
�
Ybert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/StopGradientStopGradientQbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/mean*
T0*,
_output_shapes
:����������
�
^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/SquaredDifferenceSquaredDifference>bert_1/encoder/transformer/group_0/layer_0/inner_group_0/add_1Ybert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/StopGradient*-
_output_shapes
:�����������*
T0
�
gbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
�
Ubert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/varianceMean^bert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/SquaredDifferencegbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/variance/reduction_indices*,
_output_shapes
:����������*
	keep_dims(*
T0
�
Tbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *̼�+*
dtype0
�
Rbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/addAddUbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/varianceTbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/add/y*,
_output_shapes
:����������*
T0
�
Tbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/RsqrtRsqrtRbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/add*
T0*,
_output_shapes
:����������
�
Rbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/mulMulTbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/RsqrtEbert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma/read*
T0*-
_output_shapes
:�����������
�
Tbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/mul_1Mul>bert_1/encoder/transformer/group_0/layer_0/inner_group_0/add_1Rbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/mul*
T0*-
_output_shapes
:�����������
�
Tbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/mul_2MulQbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/moments/meanRbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/mul*
T0*-
_output_shapes
:�����������
�
Rbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/subSubDbert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta/readTbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/mul_2*-
_output_shapes
:�����������*
T0
�
Tbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/add_1AddTbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/mul_1Rbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/sub*
T0*-
_output_shapes
:�����������
v
!bert_1/pooler/strided_slice/stackConst*
_output_shapes
:*!
valueB"            *
dtype0
x
#bert_1/pooler/strided_slice/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
x
#bert_1/pooler/strided_slice/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
�
bert_1/pooler/strided_sliceStridedSliceTbert_1/encoder/transformer/group_0/layer_0/inner_group_0/LayerNorm_1/batchnorm/add_1!bert_1/pooler/strided_slice/stack#bert_1/pooler/strided_slice/stack_1#bert_1/pooler/strided_slice/stack_2*
T0*
Index0*

begin_mask*
end_mask*,
_output_shapes
:����������
�
bert_1/pooler/SqueezeSqueezebert_1/pooler/strided_slice*
squeeze_dims
*
T0*(
_output_shapes
:����������
�
bert_1/pooler/dense/MatMulMatMulbert_1/pooler/Squeezebert/pooler/dense/kernel/read*(
_output_shapes
:����������*
T0
�
bert_1/pooler/dense/BiasAddBiasAddbert_1/pooler/dense/MatMulbert/pooler/dense/bias/read*
T0*(
_output_shapes
:����������
p
bert_1/pooler/dense/TanhTanhbert_1/pooler/dense/BiasAdd*
T0*(
_output_shapes
:����������
m
loss/l2_normalize/SquareSquarebert/pooler/dense/Tanh*(
_output_shapes
:����������*
T0
r
'loss/l2_normalize/Sum/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
loss/l2_normalize/SumSumloss/l2_normalize/Square'loss/l2_normalize/Sum/reduction_indices*
	keep_dims(*
T0*'
_output_shapes
:���������
`
loss/l2_normalize/Maximum/yConst*
valueB
 *̼�+*
dtype0*
_output_shapes
: 
�
loss/l2_normalize/MaximumMaximumloss/l2_normalize/Sumloss/l2_normalize/Maximum/y*
T0*'
_output_shapes
:���������
m
loss/l2_normalize/RsqrtRsqrtloss/l2_normalize/Maximum*'
_output_shapes
:���������*
T0
|
loss/l2_normalizeMulbert/pooler/dense/Tanhloss/l2_normalize/Rsqrt*(
_output_shapes
:����������*
T0
q
loss/l2_normalize_1/SquareSquarebert_1/pooler/dense/Tanh*(
_output_shapes
:����������*
T0
t
)loss/l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
loss/l2_normalize_1/SumSumloss/l2_normalize_1/Square)loss/l2_normalize_1/Sum/reduction_indices*
	keep_dims(*
T0*'
_output_shapes
:���������
b
loss/l2_normalize_1/Maximum/yConst*
valueB
 *̼�+*
dtype0*
_output_shapes
: 
�
loss/l2_normalize_1/MaximumMaximumloss/l2_normalize_1/Sumloss/l2_normalize_1/Maximum/y*
T0*'
_output_shapes
:���������
q
loss/l2_normalize_1/RsqrtRsqrtloss/l2_normalize_1/Maximum*'
_output_shapes
:���������*
T0
�
loss/l2_normalize_1Mulbert_1/pooler/dense/Tanhloss/l2_normalize_1/Rsqrt*(
_output_shapes
:����������*
T0
j
loss/mulMulloss/l2_normalizeloss/l2_normalize_1*
T0*(
_output_shapes
:����������
e
loss/Sum/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
c
loss/SumSumloss/mulloss/Sum/reduction_indices*
T0*#
_output_shapes
:���������
S
loss/subSubloss/Sum
unique_ids*
T0*#
_output_shapes
:���������
M
loss/SquareSquareloss/sub*#
_output_shapes
:���������*
T0
f
loss/Mean/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
\
	loss/MeanMeanloss/Squareloss/Mean/reduction_indices*
T0*
_output_shapes
: 
�
checkpoint_initializer/prefixConst"/device:CPU:0*H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614*
dtype0*
_output_shapes
: 
�
#checkpoint_initializer/tensor_namesConst"/device:CPU:0*3
value*B(Bbert/embeddings/LayerNorm/beta*
dtype0*
_output_shapes
:

'checkpoint_initializer/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
checkpoint_initializer	RestoreV2checkpoint_initializer/prefix#checkpoint_initializer/tensor_names'checkpoint_initializer/shape_and_slices"/device:CPU:0*
_output_shapes	
:�*
dtypes
2
�
AssignAssignbert/embeddings/LayerNorm/betacheckpoint_initializer*1
_class'
%#loc:@bert/embeddings/LayerNorm/beta*
_output_shapes	
:�*
T0
�
checkpoint_initializer_1/prefixConst"/device:CPU:0*H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614*
dtype0*
_output_shapes
: 
�
%checkpoint_initializer_1/tensor_namesConst"/device:CPU:0*4
value+B)Bbert/embeddings/LayerNorm/gamma*
dtype0*
_output_shapes
:
�
)checkpoint_initializer_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
checkpoint_initializer_1	RestoreV2checkpoint_initializer_1/prefix%checkpoint_initializer_1/tensor_names)checkpoint_initializer_1/shape_and_slices"/device:CPU:0*
_output_shapes	
:�*
dtypes
2
�
Assign_1Assignbert/embeddings/LayerNorm/gammacheckpoint_initializer_1*
T0*2
_class(
&$loc:@bert/embeddings/LayerNorm/gamma*
_output_shapes	
:�
�
checkpoint_initializer_2/prefixConst"/device:CPU:0*
dtype0*
_output_shapes
: *H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614
�
%checkpoint_initializer_2/tensor_namesConst"/device:CPU:0*8
value/B-B#bert/embeddings/position_embeddings*
dtype0*
_output_shapes
:
�
)checkpoint_initializer_2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0
�
checkpoint_initializer_2	RestoreV2checkpoint_initializer_2/prefix%checkpoint_initializer_2/tensor_names)checkpoint_initializer_2/shape_and_slices"/device:CPU:0* 
_output_shapes
:
��*
dtypes
2
�
Assign_2Assign#bert/embeddings/position_embeddingscheckpoint_initializer_2* 
_output_shapes
:
��*
T0*6
_class,
*(loc:@bert/embeddings/position_embeddings
�
checkpoint_initializer_3/prefixConst"/device:CPU:0*H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614*
dtype0*
_output_shapes
: 
�
%checkpoint_initializer_3/tensor_namesConst"/device:CPU:0*
_output_shapes
:*:
value1B/B%bert/embeddings/token_type_embeddings*
dtype0
�
)checkpoint_initializer_3/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
checkpoint_initializer_3	RestoreV2checkpoint_initializer_3/prefix%checkpoint_initializer_3/tensor_names)checkpoint_initializer_3/shape_and_slices"/device:CPU:0*
_output_shapes
:	�*
dtypes
2
�
Assign_3Assign%bert/embeddings/token_type_embeddingscheckpoint_initializer_3*
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
_output_shapes
:	�
�
checkpoint_initializer_4/prefixConst"/device:CPU:0*H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614*
dtype0*
_output_shapes
: 
�
%checkpoint_initializer_4/tensor_namesConst"/device:CPU:0*4
value+B)Bbert/embeddings/word_embeddings*
dtype0*
_output_shapes
:
�
)checkpoint_initializer_4/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
checkpoint_initializer_4	RestoreV2checkpoint_initializer_4/prefix%checkpoint_initializer_4/tensor_names)checkpoint_initializer_4/shape_and_slices"/device:CPU:0*!
_output_shapes
:���*
dtypes
2
�
Assign_4Assignbert/embeddings/word_embeddingscheckpoint_initializer_4*
T0*2
_class(
&$loc:@bert/embeddings/word_embeddings*!
_output_shapes
:���
�
checkpoint_initializer_5/prefixConst"/device:CPU:0*H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614*
dtype0*
_output_shapes
: 
�
%checkpoint_initializer_5/tensor_namesConst"/device:CPU:0*B
value9B7B-bert/encoder/embedding_hidden_mapping_in/bias*
dtype0*
_output_shapes
:
�
)checkpoint_initializer_5/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
checkpoint_initializer_5	RestoreV2checkpoint_initializer_5/prefix%checkpoint_initializer_5/tensor_names)checkpoint_initializer_5/shape_and_slices"/device:CPU:0*
_output_shapes	
:�*
dtypes
2
�
Assign_5Assign-bert/encoder/embedding_hidden_mapping_in/biascheckpoint_initializer_5*@
_class6
42loc:@bert/encoder/embedding_hidden_mapping_in/bias*
_output_shapes	
:�*
T0
�
checkpoint_initializer_6/prefixConst"/device:CPU:0*H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614*
dtype0*
_output_shapes
: 
�
%checkpoint_initializer_6/tensor_namesConst"/device:CPU:0*D
value;B9B/bert/encoder/embedding_hidden_mapping_in/kernel*
dtype0*
_output_shapes
:
�
)checkpoint_initializer_6/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0
�
checkpoint_initializer_6	RestoreV2checkpoint_initializer_6/prefix%checkpoint_initializer_6/tensor_names)checkpoint_initializer_6/shape_and_slices"/device:CPU:0* 
_output_shapes
:
��*
dtypes
2
�
Assign_6Assign/bert/encoder/embedding_hidden_mapping_in/kernelcheckpoint_initializer_6* 
_output_shapes
:
��*
T0*B
_class8
64loc:@bert/encoder/embedding_hidden_mapping_in/kernel
�
checkpoint_initializer_7/prefixConst"/device:CPU:0*H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614*
dtype0*
_output_shapes
: 
�
%checkpoint_initializer_7/tensor_namesConst"/device:CPU:0*R
valueIBGB=bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta*
dtype0*
_output_shapes
:
�
)checkpoint_initializer_7/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
checkpoint_initializer_7	RestoreV2checkpoint_initializer_7/prefix%checkpoint_initializer_7/tensor_names)checkpoint_initializer_7/shape_and_slices"/device:CPU:0*
_output_shapes	
:�*
dtypes
2
�
Assign_7Assign=bert/encoder/transformer/group_0/inner_group_0/LayerNorm/betacheckpoint_initializer_7*
T0*P
_classF
DBloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta*
_output_shapes	
:�
�
checkpoint_initializer_8/prefixConst"/device:CPU:0*H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614*
dtype0*
_output_shapes
: 
�
%checkpoint_initializer_8/tensor_namesConst"/device:CPU:0*S
valueJBHB>bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma*
dtype0*
_output_shapes
:
�
)checkpoint_initializer_8/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
checkpoint_initializer_8	RestoreV2checkpoint_initializer_8/prefix%checkpoint_initializer_8/tensor_names)checkpoint_initializer_8/shape_and_slices"/device:CPU:0*
_output_shapes	
:�*
dtypes
2
�
Assign_8Assign>bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gammacheckpoint_initializer_8*
T0*Q
_classG
ECloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma*
_output_shapes	
:�
�
checkpoint_initializer_9/prefixConst"/device:CPU:0*H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614*
dtype0*
_output_shapes
: 
�
%checkpoint_initializer_9/tensor_namesConst"/device:CPU:0*T
valueKBIB?bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta*
dtype0*
_output_shapes
:
�
)checkpoint_initializer_9/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0
�
checkpoint_initializer_9	RestoreV2checkpoint_initializer_9/prefix%checkpoint_initializer_9/tensor_names)checkpoint_initializer_9/shape_and_slices"/device:CPU:0*
_output_shapes	
:�*
dtypes
2
�
Assign_9Assign?bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/betacheckpoint_initializer_9*
T0*R
_classH
FDloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta*
_output_shapes	
:�
�
 checkpoint_initializer_10/prefixConst"/device:CPU:0*
dtype0*
_output_shapes
: *H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614
�
&checkpoint_initializer_10/tensor_namesConst"/device:CPU:0*U
valueLBJB@bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma*
dtype0*
_output_shapes
:
�
*checkpoint_initializer_10/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
checkpoint_initializer_10	RestoreV2 checkpoint_initializer_10/prefix&checkpoint_initializer_10/tensor_names*checkpoint_initializer_10/shape_and_slices"/device:CPU:0*
_output_shapes	
:�*
dtypes
2
�
	Assign_10Assign@bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gammacheckpoint_initializer_10*
T0*S
_classI
GEloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma*
_output_shapes	
:�
�
 checkpoint_initializer_11/prefixConst"/device:CPU:0*H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614*
dtype0*
_output_shapes
: 
�
&checkpoint_initializer_11/tensor_namesConst"/device:CPU:0*a
valueXBVBLbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias*
dtype0*
_output_shapes
:
�
*checkpoint_initializer_11/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
checkpoint_initializer_11	RestoreV2 checkpoint_initializer_11/prefix&checkpoint_initializer_11/tensor_names*checkpoint_initializer_11/shape_and_slices"/device:CPU:0*
_output_shapes	
:�*
dtypes
2
�
	Assign_11AssignLbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/biascheckpoint_initializer_11*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias*
_output_shapes	
:�
�
 checkpoint_initializer_12/prefixConst"/device:CPU:0*H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614*
dtype0*
_output_shapes
: 
�
&checkpoint_initializer_12/tensor_namesConst"/device:CPU:0*c
valueZBXBNbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel*
dtype0*
_output_shapes
:
�
*checkpoint_initializer_12/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B 
�
checkpoint_initializer_12	RestoreV2 checkpoint_initializer_12/prefix&checkpoint_initializer_12/tensor_names*checkpoint_initializer_12/shape_and_slices"/device:CPU:0* 
_output_shapes
:
��*
dtypes
2
�
	Assign_12AssignNbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernelcheckpoint_initializer_12*
T0*a
_classW
USloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel* 
_output_shapes
:
��
�
 checkpoint_initializer_13/prefixConst"/device:CPU:0*H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614*
dtype0*
_output_shapes
: 
�
&checkpoint_initializer_13/tensor_namesConst"/device:CPU:0*]
valueTBRBHbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias*
dtype0*
_output_shapes
:
�
*checkpoint_initializer_13/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
checkpoint_initializer_13	RestoreV2 checkpoint_initializer_13/prefix&checkpoint_initializer_13/tensor_names*checkpoint_initializer_13/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes	
:�
�
	Assign_13AssignHbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/biascheckpoint_initializer_13*
_output_shapes	
:�*
T0*[
_classQ
OMloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias
�
 checkpoint_initializer_14/prefixConst"/device:CPU:0*H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614*
dtype0*
_output_shapes
: 
�
&checkpoint_initializer_14/tensor_namesConst"/device:CPU:0*_
valueVBTBJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel*
dtype0*
_output_shapes
:
�
*checkpoint_initializer_14/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0
�
checkpoint_initializer_14	RestoreV2 checkpoint_initializer_14/prefix&checkpoint_initializer_14/tensor_names*checkpoint_initializer_14/shape_and_slices"/device:CPU:0*
dtypes
2* 
_output_shapes
:
��
�
	Assign_14AssignJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernelcheckpoint_initializer_14*
T0*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel* 
_output_shapes
:
��
�
 checkpoint_initializer_15/prefixConst"/device:CPU:0*H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614*
dtype0*
_output_shapes
: 
�
&checkpoint_initializer_15/tensor_namesConst"/device:CPU:0*_
valueVBTBJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias*
dtype0*
_output_shapes
:
�
*checkpoint_initializer_15/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
checkpoint_initializer_15	RestoreV2 checkpoint_initializer_15/prefix&checkpoint_initializer_15/tensor_names*checkpoint_initializer_15/shape_and_slices"/device:CPU:0*
_output_shapes	
:�*
dtypes
2
�
	Assign_15AssignJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/biascheckpoint_initializer_15*
T0*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias*
_output_shapes	
:�
�
 checkpoint_initializer_16/prefixConst"/device:CPU:0*H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614*
dtype0*
_output_shapes
: 
�
&checkpoint_initializer_16/tensor_namesConst"/device:CPU:0*a
valueXBVBLbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel*
dtype0*
_output_shapes
:
�
*checkpoint_initializer_16/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
checkpoint_initializer_16	RestoreV2 checkpoint_initializer_16/prefix&checkpoint_initializer_16/tensor_names*checkpoint_initializer_16/shape_and_slices"/device:CPU:0* 
_output_shapes
:
��*
dtypes
2
�
	Assign_16AssignLbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernelcheckpoint_initializer_16* 
_output_shapes
:
��*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel
�
 checkpoint_initializer_17/prefixConst"/device:CPU:0*
dtype0*
_output_shapes
: *H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614
�
&checkpoint_initializer_17/tensor_namesConst"/device:CPU:0*_
valueVBTBJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias*
dtype0*
_output_shapes
:
�
*checkpoint_initializer_17/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
checkpoint_initializer_17	RestoreV2 checkpoint_initializer_17/prefix&checkpoint_initializer_17/tensor_names*checkpoint_initializer_17/shape_and_slices"/device:CPU:0*
_output_shapes	
:�*
dtypes
2
�
	Assign_17AssignJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/biascheckpoint_initializer_17*
_output_shapes	
:�*
T0*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias
�
 checkpoint_initializer_18/prefixConst"/device:CPU:0*H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614*
dtype0*
_output_shapes
: 
�
&checkpoint_initializer_18/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*a
valueXBVBLbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel
�
*checkpoint_initializer_18/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
checkpoint_initializer_18	RestoreV2 checkpoint_initializer_18/prefix&checkpoint_initializer_18/tensor_names*checkpoint_initializer_18/shape_and_slices"/device:CPU:0*
dtypes
2* 
_output_shapes
:
��
�
	Assign_18AssignLbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernelcheckpoint_initializer_18*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel* 
_output_shapes
:
��
�
 checkpoint_initializer_19/prefixConst"/device:CPU:0*H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614*
dtype0*
_output_shapes
: 
�
&checkpoint_initializer_19/tensor_namesConst"/device:CPU:0*a
valueXBVBLbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias*
dtype0*
_output_shapes
:
�
*checkpoint_initializer_19/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0
�
checkpoint_initializer_19	RestoreV2 checkpoint_initializer_19/prefix&checkpoint_initializer_19/tensor_names*checkpoint_initializer_19/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes	
:�	
�
	Assign_19AssignLbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/biascheckpoint_initializer_19*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias*
_output_shapes	
:�	
�
 checkpoint_initializer_20/prefixConst"/device:CPU:0*H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614*
dtype0*
_output_shapes
: 
�
&checkpoint_initializer_20/tensor_namesConst"/device:CPU:0*c
valueZBXBNbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel*
dtype0*
_output_shapes
:
�
*checkpoint_initializer_20/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
checkpoint_initializer_20	RestoreV2 checkpoint_initializer_20/prefix&checkpoint_initializer_20/tensor_names*checkpoint_initializer_20/shape_and_slices"/device:CPU:0* 
_output_shapes
:
��	*
dtypes
2
�
	Assign_20AssignNbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernelcheckpoint_initializer_20* 
_output_shapes
:
��	*
T0*a
_classW
USloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel
�
 checkpoint_initializer_21/prefixConst"/device:CPU:0*H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614*
dtype0*
_output_shapes
: 
�
&checkpoint_initializer_21/tensor_namesConst"/device:CPU:0*h
value_B]BSbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias*
dtype0*
_output_shapes
:
�
*checkpoint_initializer_21/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
checkpoint_initializer_21	RestoreV2 checkpoint_initializer_21/prefix&checkpoint_initializer_21/tensor_names*checkpoint_initializer_21/shape_and_slices"/device:CPU:0*
_output_shapes	
:�*
dtypes
2
�
	Assign_21AssignSbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/biascheckpoint_initializer_21*
T0*f
_class\
ZXloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias*
_output_shapes	
:�
�
 checkpoint_initializer_22/prefixConst"/device:CPU:0*H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614*
dtype0*
_output_shapes
: 
�
&checkpoint_initializer_22/tensor_namesConst"/device:CPU:0*j
valueaB_BUbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel*
dtype0*
_output_shapes
:
�
*checkpoint_initializer_22/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0
�
checkpoint_initializer_22	RestoreV2 checkpoint_initializer_22/prefix&checkpoint_initializer_22/tensor_names*checkpoint_initializer_22/shape_and_slices"/device:CPU:0* 
_output_shapes
:
�	�*
dtypes
2
�
	Assign_22AssignUbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernelcheckpoint_initializer_22*
T0*h
_class^
\Zloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel* 
_output_shapes
:
�	�
�
 checkpoint_initializer_23/prefixConst"/device:CPU:0*H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614*
dtype0*
_output_shapes
: 
�
&checkpoint_initializer_23/tensor_namesConst"/device:CPU:0*+
value"B Bbert/pooler/dense/bias*
dtype0*
_output_shapes
:
�
*checkpoint_initializer_23/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
�
checkpoint_initializer_23	RestoreV2 checkpoint_initializer_23/prefix&checkpoint_initializer_23/tensor_names*checkpoint_initializer_23/shape_and_slices"/device:CPU:0*
_output_shapes	
:�*
dtypes
2
�
	Assign_23Assignbert/pooler/dense/biascheckpoint_initializer_23*
T0*)
_class
loc:@bert/pooler/dense/bias*
_output_shapes	
:�
�
 checkpoint_initializer_24/prefixConst"/device:CPU:0*H
value?B= B7./albert_lcqmc_tiny_google_checkpoints/model.ckpt-74614*
dtype0*
_output_shapes
: 
�
&checkpoint_initializer_24/tensor_namesConst"/device:CPU:0*-
value$B"Bbert/pooler/dense/kernel*
dtype0*
_output_shapes
:
�
*checkpoint_initializer_24/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0
�
checkpoint_initializer_24	RestoreV2 checkpoint_initializer_24/prefix&checkpoint_initializer_24/tensor_names*checkpoint_initializer_24/shape_and_slices"/device:CPU:0* 
_output_shapes
:
��*
dtypes
2
�
	Assign_24Assignbert/pooler/dense/kernelcheckpoint_initializer_24*
T0*+
_class!
loc:@bert/pooler/dense/kernel* 
_output_shapes
:
��

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_e7e444d5520e4c83bc59b374d1746e24/part*
dtype0*
_output_shapes
: 
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*�
value�B�Bbert/embeddings/LayerNorm/betaBbert/embeddings/LayerNorm/gammaB#bert/embeddings/position_embeddingsB%bert/embeddings/token_type_embeddingsBbert/embeddings/word_embeddingsB-bert/encoder/embedding_hidden_mapping_in/biasB/bert/encoder/embedding_hidden_mapping_in/kernelB=bert/encoder/transformer/group_0/inner_group_0/LayerNorm/betaB>bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gammaB?bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/betaB@bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gammaBLbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/biasBNbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernelBHbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/biasBJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernelBJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/biasBLbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernelBJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/biasBLbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernelBLbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/biasBNbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernelBSbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/biasBUbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernelBbert/pooler/dense/biasBbert/pooler/dense/kernelBglobal_step
�
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbert/embeddings/LayerNorm/betabert/embeddings/LayerNorm/gamma#bert/embeddings/position_embeddings%bert/embeddings/token_type_embeddingsbert/embeddings/word_embeddings-bert/encoder/embedding_hidden_mapping_in/bias/bert/encoder/embedding_hidden_mapping_in/kernel=bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta>bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma?bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta@bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gammaLbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/biasNbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernelHbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/biasJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernelJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/biasLbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernelJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/biasLbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernelLbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/biasNbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernelSbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/biasUbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernelbert/pooler/dense/biasbert/pooler/dense/kernelglobal_step/Read/ReadVariableOp"/device:CPU:0*(
dtypes
2	
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*
N*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
�
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�Bbert/embeddings/LayerNorm/betaBbert/embeddings/LayerNorm/gammaB#bert/embeddings/position_embeddingsB%bert/embeddings/token_type_embeddingsBbert/embeddings/word_embeddingsB-bert/encoder/embedding_hidden_mapping_in/biasB/bert/encoder/embedding_hidden_mapping_in/kernelB=bert/encoder/transformer/group_0/inner_group_0/LayerNorm/betaB>bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gammaB?bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/betaB@bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gammaBLbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/biasBNbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernelBHbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/biasBJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernelBJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/biasBLbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernelBJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/biasBLbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernelBLbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/biasBNbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernelBSbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/biasBUbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernelBbert/pooler/dense/biasBbert/pooler/dense/kernelBglobal_step*
dtype0*
_output_shapes
:
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	
�
save/AssignAssignbert/embeddings/LayerNorm/betasave/RestoreV2*
_output_shapes	
:�*
T0*1
_class'
%#loc:@bert/embeddings/LayerNorm/beta
�
save/Assign_1Assignbert/embeddings/LayerNorm/gammasave/RestoreV2:1*
T0*2
_class(
&$loc:@bert/embeddings/LayerNorm/gamma*
_output_shapes	
:�
�
save/Assign_2Assign#bert/embeddings/position_embeddingssave/RestoreV2:2*
T0*6
_class,
*(loc:@bert/embeddings/position_embeddings* 
_output_shapes
:
��
�
save/Assign_3Assign%bert/embeddings/token_type_embeddingssave/RestoreV2:3*
_output_shapes
:	�*
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings
�
save/Assign_4Assignbert/embeddings/word_embeddingssave/RestoreV2:4*
T0*2
_class(
&$loc:@bert/embeddings/word_embeddings*!
_output_shapes
:���
�
save/Assign_5Assign-bert/encoder/embedding_hidden_mapping_in/biassave/RestoreV2:5*
T0*@
_class6
42loc:@bert/encoder/embedding_hidden_mapping_in/bias*
_output_shapes	
:�
�
save/Assign_6Assign/bert/encoder/embedding_hidden_mapping_in/kernelsave/RestoreV2:6*
T0*B
_class8
64loc:@bert/encoder/embedding_hidden_mapping_in/kernel* 
_output_shapes
:
��
�
save/Assign_7Assign=bert/encoder/transformer/group_0/inner_group_0/LayerNorm/betasave/RestoreV2:7*P
_classF
DBloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta*
_output_shapes	
:�*
T0
�
save/Assign_8Assign>bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gammasave/RestoreV2:8*
T0*Q
_classG
ECloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma*
_output_shapes	
:�
�
save/Assign_9Assign?bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/betasave/RestoreV2:9*
_output_shapes	
:�*
T0*R
_classH
FDloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta
�
save/Assign_10Assign@bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gammasave/RestoreV2:10*
T0*S
_classI
GEloc:@bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma*
_output_shapes	
:�
�
save/Assign_11AssignLbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/biassave/RestoreV2:11*
_output_shapes	
:�*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias
�
save/Assign_12AssignNbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernelsave/RestoreV2:12*a
_classW
USloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel* 
_output_shapes
:
��*
T0
�
save/Assign_13AssignHbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/biassave/RestoreV2:13*
_output_shapes	
:�*
T0*[
_classQ
OMloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias
�
save/Assign_14AssignJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernelsave/RestoreV2:14*
T0*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel* 
_output_shapes
:
��
�
save/Assign_15AssignJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/biassave/RestoreV2:15*
_output_shapes	
:�*
T0*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias
�
save/Assign_16AssignLbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernelsave/RestoreV2:16*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel* 
_output_shapes
:
��
�
save/Assign_17AssignJbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/biassave/RestoreV2:17*
T0*]
_classS
QOloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias*
_output_shapes	
:�
�
save/Assign_18AssignLbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernelsave/RestoreV2:18*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel* 
_output_shapes
:
��
�
save/Assign_19AssignLbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/biassave/RestoreV2:19*
T0*_
_classU
SQloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias*
_output_shapes	
:�	
�
save/Assign_20AssignNbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernelsave/RestoreV2:20* 
_output_shapes
:
��	*
T0*a
_classW
USloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel
�
save/Assign_21AssignSbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/biassave/RestoreV2:21*
_output_shapes	
:�*
T0*f
_class\
ZXloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias
�
save/Assign_22AssignUbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernelsave/RestoreV2:22*
T0*h
_class^
\Zloc:@bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel* 
_output_shapes
:
�	�
�
save/Assign_23Assignbert/pooler/dense/biassave/RestoreV2:23*
_output_shapes	
:�*
T0*)
_class
loc:@bert/pooler/dense/bias
�
save/Assign_24Assignbert/pooler/dense/kernelsave/RestoreV2:24* 
_output_shapes
:
��*
T0*+
_class!
loc:@bert/pooler/dense/kernel
Q
save/Identity_1Identitysave/RestoreV2:25*
T0	*
_output_shapes
:
T
save/AssignVariableOpAssignVariableOpglobal_stepsave/Identity_1*
dtype0	
�
save/restore_shardNoOp^save/Assign^save/AssignVariableOp^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"|
global_stepmk
i
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0"�#
	variables�#�#
i
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0
u
!bert/embeddings/word_embeddings:0
Assign_4:0&bert/embeddings/word_embeddings/read:02checkpoint_initializer_4:08
�
'bert/embeddings/token_type_embeddings:0
Assign_3:0,bert/embeddings/token_type_embeddings/read:02checkpoint_initializer_3:08
}
%bert/embeddings/position_embeddings:0
Assign_2:0*bert/embeddings/position_embeddings/read:02checkpoint_initializer_2:08
o
 bert/embeddings/LayerNorm/beta:0Assign:0%bert/embeddings/LayerNorm/beta/read:02checkpoint_initializer:08
u
!bert/embeddings/LayerNorm/gamma:0
Assign_1:0&bert/embeddings/LayerNorm/gamma/read:02checkpoint_initializer_1:08
�
1bert/encoder/embedding_hidden_mapping_in/kernel:0
Assign_6:06bert/encoder/embedding_hidden_mapping_in/kernel/read:02checkpoint_initializer_6:08
�
/bert/encoder/embedding_hidden_mapping_in/bias:0
Assign_5:04bert/encoder/embedding_hidden_mapping_in/bias/read:02checkpoint_initializer_5:08
�
Nbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel:0Assign_16:0Sbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel/read:02checkpoint_initializer_16:08
�
Lbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias:0Assign_15:0Qbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias/read:02checkpoint_initializer_15:08
�
Lbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel:0Assign_14:0Qbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel/read:02checkpoint_initializer_14:08
�
Jbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias:0Assign_13:0Obert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias/read:02checkpoint_initializer_13:08
�
Nbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel:0Assign_18:0Sbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel/read:02checkpoint_initializer_18:08
�
Lbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias:0Assign_17:0Qbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias/read:02checkpoint_initializer_17:08
�
Pbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel:0Assign_12:0Ubert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel/read:02checkpoint_initializer_12:08
�
Nbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias:0Assign_11:0Sbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias/read:02checkpoint_initializer_11:08
�
?bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta:0
Assign_7:0Dbert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta/read:02checkpoint_initializer_7:08
�
@bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma:0
Assign_8:0Ebert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma/read:02checkpoint_initializer_8:08
�
Pbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel:0Assign_20:0Ubert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel/read:02checkpoint_initializer_20:08
�
Nbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias:0Assign_19:0Sbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias/read:02checkpoint_initializer_19:08
�
Wbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel:0Assign_22:0\bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel/read:02checkpoint_initializer_22:08
�
Ubert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias:0Assign_21:0Zbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias/read:02checkpoint_initializer_21:08
�
Abert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta:0
Assign_9:0Fbert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta/read:02checkpoint_initializer_9:08
�
Bbert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma:0Assign_10:0Gbert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma/read:02checkpoint_initializer_10:08
i
bert/pooler/dense/kernel:0Assign_24:0bert/pooler/dense/kernel/read:02checkpoint_initializer_24:08
e
bert/pooler/dense/bias:0Assign_23:0bert/pooler/dense/bias/read:02checkpoint_initializer_23:08"%
saved_model_main_op


group_deps"�
model_variables��
o
 bert/embeddings/LayerNorm/beta:0Assign:0%bert/embeddings/LayerNorm/beta/read:02checkpoint_initializer:08
u
!bert/embeddings/LayerNorm/gamma:0
Assign_1:0&bert/embeddings/LayerNorm/gamma/read:02checkpoint_initializer_1:08
�
?bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta:0
Assign_7:0Dbert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta/read:02checkpoint_initializer_7:08
�
@bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma:0
Assign_8:0Ebert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma/read:02checkpoint_initializer_8:08
�
Abert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta:0
Assign_9:0Fbert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta/read:02checkpoint_initializer_9:08
�
Bbert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma:0Assign_10:0Gbert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma/read:02checkpoint_initializer_10:08"�"
trainable_variables�"�"
u
!bert/embeddings/word_embeddings:0
Assign_4:0&bert/embeddings/word_embeddings/read:02checkpoint_initializer_4:08
�
'bert/embeddings/token_type_embeddings:0
Assign_3:0,bert/embeddings/token_type_embeddings/read:02checkpoint_initializer_3:08
}
%bert/embeddings/position_embeddings:0
Assign_2:0*bert/embeddings/position_embeddings/read:02checkpoint_initializer_2:08
o
 bert/embeddings/LayerNorm/beta:0Assign:0%bert/embeddings/LayerNorm/beta/read:02checkpoint_initializer:08
u
!bert/embeddings/LayerNorm/gamma:0
Assign_1:0&bert/embeddings/LayerNorm/gamma/read:02checkpoint_initializer_1:08
�
1bert/encoder/embedding_hidden_mapping_in/kernel:0
Assign_6:06bert/encoder/embedding_hidden_mapping_in/kernel/read:02checkpoint_initializer_6:08
�
/bert/encoder/embedding_hidden_mapping_in/bias:0
Assign_5:04bert/encoder/embedding_hidden_mapping_in/bias/read:02checkpoint_initializer_5:08
�
Nbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel:0Assign_16:0Sbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel/read:02checkpoint_initializer_16:08
�
Lbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias:0Assign_15:0Qbert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias/read:02checkpoint_initializer_15:08
�
Lbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel:0Assign_14:0Qbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel/read:02checkpoint_initializer_14:08
�
Jbert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias:0Assign_13:0Obert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias/read:02checkpoint_initializer_13:08
�
Nbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel:0Assign_18:0Sbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel/read:02checkpoint_initializer_18:08
�
Lbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias:0Assign_17:0Qbert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias/read:02checkpoint_initializer_17:08
�
Pbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel:0Assign_12:0Ubert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel/read:02checkpoint_initializer_12:08
�
Nbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias:0Assign_11:0Sbert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias/read:02checkpoint_initializer_11:08
�
?bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta:0
Assign_7:0Dbert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta/read:02checkpoint_initializer_7:08
�
@bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma:0
Assign_8:0Ebert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma/read:02checkpoint_initializer_8:08
�
Pbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel:0Assign_20:0Ubert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel/read:02checkpoint_initializer_20:08
�
Nbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias:0Assign_19:0Sbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias/read:02checkpoint_initializer_19:08
�
Wbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel:0Assign_22:0\bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel/read:02checkpoint_initializer_22:08
�
Ubert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias:0Assign_21:0Zbert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias/read:02checkpoint_initializer_21:08
�
Abert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta:0
Assign_9:0Fbert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta/read:02checkpoint_initializer_9:08
�
Bbert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma:0Assign_10:0Gbert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma/read:02checkpoint_initializer_10:08
i
bert/pooler/dense/kernel:0Assign_24:0bert/pooler/dense/kernel/read:02checkpoint_initializer_24:08
e
bert/pooler/dense/bias:0Assign_23:0bert/pooler/dense/bias/read:02checkpoint_initializer_23:08*�
serving_default�
6
b_input_mask&
b_input_mask:0	����������
8
a_segment_ids'
a_segment_ids:0	����������
8
b_segment_ids'
b_segment_ids:0	����������
,
	label_ids
unique_ids:0���������
6
a_input_mask&
a_input_mask:0	����������
4
a_input_ids%
a_input_ids:0	����������
4
b_input_ids%
a_input_ids:0	����������B
a_output_layer0
bert/pooler/dense/Tanh:0����������tensorflow/serving/predict