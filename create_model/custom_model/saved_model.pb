??
??
.
Abs
x"T
y"T"
Ttype:

2	
E
AssignAddVariableOp
resource
value"dtype"
dtypetype?
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
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
8
DivNoNan
x"T
y"T
z"T"
Ttype:	
2
A
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
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
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
?
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
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
d
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
}
ResourceApplyGradientDescent
var

alpha"T

delta"T" 
Ttype:
2	"
use_lockingbool( ?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
x
test_in/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nametest_in/kernel
q
"test_in/kernel/Read/ReadVariableOpReadVariableOptest_in/kernel*
_output_shapes

:*
dtype0
p
test_in/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametest_in/bias
i
 test_in/bias/Read/ReadVariableOpReadVariableOptest_in/bias*
_output_shapes
:*
dtype0
z
test_out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_nametest_out/kernel
s
#test_out/kernel/Read/ReadVariableOpReadVariableOptest_out/kernel*
_output_shapes

:*
dtype0
r
test_out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametest_out/bias
k
!test_out/bias/Read/ReadVariableOpReadVariableOptest_out/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
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

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
dense_1
dense_2
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

	kernel

bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
6
iter
	decay
learning_rate
momentum
 

	0

1
2
3

	0

1
2
3
?
layer_metrics
non_trainable_variables
regularization_losses
layer_regularization_losses
trainable_variables

layers
metrics
	variables
 
MK
VARIABLE_VALUEtest_in/kernel)dense_1/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEtest_in/bias'dense_1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

	0

1

	0

1
?
layer_metrics
non_trainable_variables
regularization_losses
 layer_regularization_losses
trainable_variables

!layers
"metrics
	variables
NL
VARIABLE_VALUEtest_out/kernel)dense_2/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEtest_out/bias'dense_2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
#layer_metrics
$non_trainable_variables
regularization_losses
%layer_regularization_losses
trainable_variables

&layers
'metrics
	variables
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1

(0
)1
 
 
 
 
 
 
 
 
 
 
4
	*total
	+count
,	variables
-	keras_api
D
	.total
	/count
0
_fn_kwargs
1	variables
2	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

,	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

1	variables
\
pred_inputsPlaceholder*
_output_shapes

:*
dtype0*
shape
:
?
StatefulPartitionedCallStatefulPartitionedCallpred_inputstest_in/kerneltest_in/biastest_out/kerneltest_out/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_2230
e
train_training_inputPlaceholder*
_output_shapes

:*
dtype0*
shape
:
f
train_training_targetPlaceholder*
_output_shapes

:*
dtype0*
shape
:
?
StatefulPartitionedCall_1StatefulPartitionedCalltrain_training_inputtrain_training_targettest_in/kerneltest_in/biastest_out/kerneltest_out/biastotalcountSGD/learning_rateSGD/momentumSGD/itertotal_1count_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: *$
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_2215
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename"test_in/kernel/Read/ReadVariableOp test_in/bias/Read/ReadVariableOp#test_out/kernel/Read/ReadVariableOp!test_out/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2	*
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
GPU 2J 8? *&
f!R
__inference__traced_save_2631
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenametest_in/kerneltest_in/biastest_out/kerneltest_out/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcounttotal_1count_1*
Tin
2*
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
GPU 2J 8? *)
f$R"
 __inference__traced_restore_2677??
?	
?
A__inference_test_in_layer_call_and_return_conditional_losses_2550

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?4
?
 __inference__traced_restore_2677
file_prefix1
assignvariableop_test_in_kernel:-
assignvariableop_1_test_in_bias:4
"assignvariableop_2_test_out_kernel:.
 assignvariableop_3_test_out_bias:%
assignvariableop_4_sgd_iter:	 &
assignvariableop_5_sgd_decay: .
$assignvariableop_6_sgd_learning_rate: )
assignvariableop_7_sgd_momentum: "
assignvariableop_8_total: "
assignvariableop_9_count: %
assignvariableop_10_total_1: %
assignvariableop_11_count_1: 
identity_13??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)dense_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dense_1/bias/.ATTRIBUTES/VARIABLE_VALUEB)dense_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dense_2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_test_in_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_test_in_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_test_out_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_test_out_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_sgd_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_sgd_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp$assignvariableop_6_sgd_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_sgd_momentumIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_totalIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_countIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12?
Identity_13IdentityIdentity_12:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_13"#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
B__inference_test_out_layer_call_and_return_conditional_losses_2569

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_call_2056

inputs8
&test_in_matmul_readvariableop_resource:5
'test_in_biasadd_readvariableop_resource:9
'test_out_matmul_readvariableop_resource:6
(test_out_biasadd_readvariableop_resource:
identity??test_in/BiasAdd/ReadVariableOp?test_in/MatMul/ReadVariableOp?test_out/BiasAdd/ReadVariableOp?test_out/MatMul/ReadVariableOp?
test_in/MatMul/ReadVariableOpReadVariableOp&test_in_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
test_in/MatMul/ReadVariableOp?
test_in/MatMulMatMulinputs%test_in/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_in/MatMul?
test_in/BiasAdd/ReadVariableOpReadVariableOp'test_in_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
test_in/BiasAdd/ReadVariableOp?
test_in/BiasAddBiasAddtest_in/MatMul:product:0&test_in/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_in/BiasAdd?
test_out/MatMul/ReadVariableOpReadVariableOp'test_out_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
test_out/MatMul/ReadVariableOp?
test_out/MatMulMatMultest_in/BiasAdd:output:0&test_out/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_out/MatMul?
test_out/BiasAdd/ReadVariableOpReadVariableOp(test_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
test_out/BiasAdd/ReadVariableOp?
test_out/BiasAddBiasAddtest_out/MatMul:product:0'test_out/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_out/BiasAdd?
IdentityIdentitytest_out/BiasAdd:output:0^test_in/BiasAdd/ReadVariableOp^test_in/MatMul/ReadVariableOp ^test_out/BiasAdd/ReadVariableOp^test_out/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : : : 2@
test_in/BiasAdd/ReadVariableOptest_in/BiasAdd/ReadVariableOp2>
test_in/MatMul/ReadVariableOptest_in/MatMul/ReadVariableOp2B
test_out/BiasAdd/ReadVariableOptest_out/BiasAdd/ReadVariableOp2@
test_out/MatMul/ReadVariableOptest_out/MatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
?
?
&__inference_test_in_layer_call_fn_2540

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_test_in_layer_call_and_return_conditional_losses_22742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__forward_call_2489
inputs_08
&test_in_matmul_readvariableop_resource:5
'test_in_biasadd_readvariableop_resource:9
'test_out_matmul_readvariableop_resource:6
(test_out_biasadd_readvariableop_resource:
identity"
test_out_matmul_readvariableop
test_in_biasadd!
test_in_matmul_readvariableop

inputs??test_in/BiasAdd/ReadVariableOp?test_in/MatMul/ReadVariableOp?test_out/BiasAdd/ReadVariableOp?test_out/MatMul/ReadVariableOp?
test_in/MatMul/ReadVariableOpReadVariableOp&test_in_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
test_in/MatMul/ReadVariableOp?
test_in/MatMulMatMulinputs_0%test_in/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_in/MatMul?
test_in/BiasAdd/ReadVariableOpReadVariableOp'test_in_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
test_in/BiasAdd/ReadVariableOp?
test_in/BiasAddBiasAddtest_in/MatMul:product:0&test_in/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_in/BiasAdd?
test_out/MatMul/ReadVariableOpReadVariableOp'test_out_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
test_out/MatMul/ReadVariableOp?
test_out/MatMulMatMultest_in/BiasAdd:output:0&test_out/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_out/MatMul?
test_out/BiasAdd/ReadVariableOpReadVariableOp(test_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
test_out/BiasAdd/ReadVariableOp?
test_out/BiasAddBiasAddtest_out/MatMul:product:0'test_out/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_out/BiasAdd?
IdentityIdentitytest_out/BiasAdd:output:0^test_in/BiasAdd/ReadVariableOp^test_in/MatMul/ReadVariableOp ^test_out/BiasAdd/ReadVariableOp^test_out/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0"
inputsinputs_0"+
test_in_biasaddtest_in/BiasAdd:output:0"F
test_in_matmul_readvariableop%test_in/MatMul/ReadVariableOp:value:0"H
test_out_matmul_readvariableop&test_out/MatMul/ReadVariableOp:value:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : : : *A
backward_function_name'%__inference___backward_call_2470_24902@
test_in/BiasAdd/ReadVariableOptest_in/BiasAdd/ReadVariableOp2>
test_in/MatMul/ReadVariableOptest_in/MatMul/ReadVariableOp2B
test_out/BiasAdd/ReadVariableOptest_out/BiasAdd/ReadVariableOp2@
test_out/MatMul/ReadVariableOptest_out/MatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
?

?
"__inference_signature_wrapper_2215
training_input
training_target
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7:	 
	unknown_8: 
	unknown_9: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltraining_inputtraining_targetunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: *$
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *"
fR
__inference_training_21842
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*::: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J

_output_shapes

:
(
_user_specified_nametraining_input:OK

_output_shapes

:
)
_user_specified_nametraining_target
?
?
__inference_call_2249

inputs8
&test_in_matmul_readvariableop_resource:5
'test_in_biasadd_readvariableop_resource:9
'test_out_matmul_readvariableop_resource:6
(test_out_biasadd_readvariableop_resource:
identity??test_in/BiasAdd/ReadVariableOp?test_in/MatMul/ReadVariableOp?test_out/BiasAdd/ReadVariableOp?test_out/MatMul/ReadVariableOp?
test_in/MatMul/ReadVariableOpReadVariableOp&test_in_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
test_in/MatMul/ReadVariableOp?
test_in/MatMulMatMulinputs%test_in/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
test_in/MatMul?
test_in/BiasAdd/ReadVariableOpReadVariableOp'test_in_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
test_in/BiasAdd/ReadVariableOp?
test_in/BiasAddBiasAddtest_in/MatMul:product:0&test_in/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
test_in/BiasAdd?
test_out/MatMul/ReadVariableOpReadVariableOp'test_out_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
test_out/MatMul/ReadVariableOp?
test_out/MatMulMatMultest_in/BiasAdd:output:0&test_out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
test_out/MatMul?
test_out/BiasAdd/ReadVariableOpReadVariableOp(test_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
test_out/BiasAdd/ReadVariableOp?
test_out/BiasAddBiasAddtest_out/MatMul:product:0'test_out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
test_out/BiasAdd?
IdentityIdentitytest_out/BiasAdd:output:0^test_in/BiasAdd/ReadVariableOp^test_in/MatMul/ReadVariableOp ^test_out/BiasAdd/ReadVariableOp^test_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2@
test_in/BiasAdd/ReadVariableOptest_in/BiasAdd/ReadVariableOp2>
test_in/MatMul/ReadVariableOptest_in/MatMul/ReadVariableOp2B
test_out/BiasAdd/ReadVariableOptest_out/BiasAdd/ReadVariableOp2@
test_out/MatMul/ReadVariableOptest_out/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?"
?
__inference__traced_save_2631
file_prefix-
)savev2_test_in_kernel_read_readvariableop+
'savev2_test_in_bias_read_readvariableop.
*savev2_test_out_kernel_read_readvariableop,
(savev2_test_out_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)dense_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dense_1/bias/.ATTRIBUTES/VARIABLE_VALUEB)dense_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dense_2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_test_in_kernel_read_readvariableop'savev2_test_in_bias_read_readvariableop*savev2_test_out_kernel_read_readvariableop(savev2_test_out_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*G
_input_shapes6
4: ::::: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference_call_148

inputs8
&test_in_matmul_readvariableop_resource:5
'test_in_biasadd_readvariableop_resource:9
'test_out_matmul_readvariableop_resource:6
(test_out_biasadd_readvariableop_resource:
identity??test_in/BiasAdd/ReadVariableOp?test_in/MatMul/ReadVariableOp?test_out/BiasAdd/ReadVariableOp?test_out/MatMul/ReadVariableOp?
test_in/MatMul/ReadVariableOpReadVariableOp&test_in_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
test_in/MatMul/ReadVariableOp?
test_in/MatMulMatMulinputs%test_in/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_in/MatMul?
test_in/BiasAdd/ReadVariableOpReadVariableOp'test_in_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
test_in/BiasAdd/ReadVariableOp?
test_in/BiasAddBiasAddtest_in/MatMul:product:0&test_in/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_in/BiasAdd?
test_out/MatMul/ReadVariableOpReadVariableOp'test_out_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
test_out/MatMul/ReadVariableOp?
test_out/MatMulMatMultest_in/BiasAdd:output:0&test_out/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_out/MatMul?
test_out/BiasAdd/ReadVariableOpReadVariableOp(test_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
test_out/BiasAdd/ReadVariableOp?
test_out/BiasAddBiasAddtest_out/MatMul:product:0'test_out/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_out/BiasAdd?
IdentityIdentitytest_out/BiasAdd:output:0^test_in/BiasAdd/ReadVariableOp^test_in/MatMul/ReadVariableOp ^test_out/BiasAdd/ReadVariableOp^test_out/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : : : 2@
test_in/BiasAdd/ReadVariableOptest_in/BiasAdd/ReadVariableOp2>
test_in/MatMul/ReadVariableOptest_in/MatMul/ReadVariableOp2B
test_out/BiasAdd/ReadVariableOptest_out/BiasAdd/ReadVariableOp2@
test_out/MatMul/ReadVariableOptest_out/MatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
?
?
'__inference_test_out_layer_call_fn_2559

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_test_out_layer_call_and_return_conditional_losses_22902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_call_2367

inputs8
&test_in_matmul_readvariableop_resource:5
'test_in_biasadd_readvariableop_resource:9
'test_out_matmul_readvariableop_resource:6
(test_out_biasadd_readvariableop_resource:
identity??test_in/BiasAdd/ReadVariableOp?test_in/MatMul/ReadVariableOp?test_out/BiasAdd/ReadVariableOp?test_out/MatMul/ReadVariableOp?
test_in/MatMul/ReadVariableOpReadVariableOp&test_in_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
test_in/MatMul/ReadVariableOp?
test_in/MatMulMatMulinputs%test_in/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
test_in/MatMul?
test_in/BiasAdd/ReadVariableOpReadVariableOp'test_in_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
test_in/BiasAdd/ReadVariableOp?
test_in/BiasAddBiasAddtest_in/MatMul:product:0&test_in/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
test_in/BiasAdd?
test_out/MatMul/ReadVariableOpReadVariableOp'test_out_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
test_out/MatMul/ReadVariableOp?
test_out/MatMulMatMultest_in/BiasAdd:output:0&test_out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
test_out/MatMul?
test_out/BiasAdd/ReadVariableOpReadVariableOp(test_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
test_out/BiasAdd/ReadVariableOp?
test_out/BiasAddBiasAddtest_out/MatMul:product:0'test_out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
test_out/BiasAdd?
IdentityIdentitytest_out/BiasAdd:output:0^test_in/BiasAdd/ReadVariableOp^test_in/MatMul/ReadVariableOp ^test_out/BiasAdd/ReadVariableOp^test_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2@
test_in/BiasAdd/ReadVariableOptest_in/BiasAdd/ReadVariableOp2>
test_in/MatMul/ReadVariableOptest_in/MatMul/ReadVariableOp2B
test_out/BiasAdd/ReadVariableOptest_out/BiasAdd/ReadVariableOp2@
test_out/MatMul/ReadVariableOptest_out/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_test_out_layer_call_and_return_conditional_losses_2290

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_call_2351

inputs8
&test_in_matmul_readvariableop_resource:5
'test_in_biasadd_readvariableop_resource:9
'test_out_matmul_readvariableop_resource:6
(test_out_biasadd_readvariableop_resource:
identity??test_in/BiasAdd/ReadVariableOp?test_in/MatMul/ReadVariableOp?test_out/BiasAdd/ReadVariableOp?test_out/MatMul/ReadVariableOp?
test_in/MatMul/ReadVariableOpReadVariableOp&test_in_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
test_in/MatMul/ReadVariableOp?
test_in/MatMulMatMulinputs%test_in/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_in/MatMul?
test_in/BiasAdd/ReadVariableOpReadVariableOp'test_in_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
test_in/BiasAdd/ReadVariableOp?
test_in/BiasAddBiasAddtest_in/MatMul:product:0&test_in/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_in/BiasAdd?
test_out/MatMul/ReadVariableOpReadVariableOp'test_out_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
test_out/MatMul/ReadVariableOp?
test_out/MatMulMatMultest_in/BiasAdd:output:0&test_out/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_out/MatMul?
test_out/BiasAdd/ReadVariableOpReadVariableOp(test_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
test_out/BiasAdd/ReadVariableOp?
test_out/BiasAddBiasAddtest_out/MatMul:product:0'test_out/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_out/BiasAdd?
IdentityIdentitytest_out/BiasAdd:output:0^test_in/BiasAdd/ReadVariableOp^test_in/MatMul/ReadVariableOp ^test_out/BiasAdd/ReadVariableOp^test_out/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : : : 2@
test_in/BiasAdd/ReadVariableOptest_in/BiasAdd/ReadVariableOp2>
test_in/MatMul/ReadVariableOptest_in/MatMul/ReadVariableOp2B
test_out/BiasAdd/ReadVariableOptest_out/BiasAdd/ReadVariableOp2@
test_out/MatMul/ReadVariableOptest_out/MatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
?
?
F__inference_custom_model_layer_call_and_return_conditional_losses_2297
input_1
test_in_2275:
test_in_2277:
test_out_2291:
test_out_2293:
identity??test_in/StatefulPartitionedCall? test_out/StatefulPartitionedCall?
test_in/StatefulPartitionedCallStatefulPartitionedCallinput_1test_in_2275test_in_2277*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_test_in_layer_call_and_return_conditional_losses_22742!
test_in/StatefulPartitionedCall?
 test_out/StatefulPartitionedCallStatefulPartitionedCall(test_in/StatefulPartitionedCall:output:0test_out_2291test_out_2293*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_test_out_layer_call_and_return_conditional_losses_22902"
 test_out/StatefulPartitionedCall?
IdentityIdentity)test_out/StatefulPartitionedCall:output:0 ^test_in/StatefulPartitionedCall!^test_out/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2B
test_in/StatefulPartitionedCalltest_in/StatefulPartitionedCall2D
 test_out/StatefulPartitionedCall test_out/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
??
?
__inference_training_2184
training_input
training_target#
custom_model_2061:
custom_model_2063:#
custom_model_2065:
custom_model_2067:&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: *
 sgd_cast_readvariableop_resource: ,
"sgd_cast_1_readvariableop_resource: .
$sgd_sgd_assignaddvariableop_resource:	 (
assignaddvariableop_2_resource: (
assignaddvariableop_3_resource: 

identity_2??AssignAddVariableOp?AssignAddVariableOp_1?AssignAddVariableOp_2?AssignAddVariableOp_3?SGD/Cast/ReadVariableOp?SGD/Cast_1/ReadVariableOp?SGD/SGD/AssignAddVariableOp?+SGD/SGD/update/ResourceApplyGradientDescent?-SGD/SGD/update_1/ResourceApplyGradientDescent?-SGD/SGD/update_2/ResourceApplyGradientDescent?-SGD/SGD/update_3/ResourceApplyGradientDescent?$custom_model/StatefulPartitionedCall?div_no_nan/ReadVariableOp?div_no_nan/ReadVariableOp_1?div_no_nan_1/ReadVariableOp?div_no_nan_1/ReadVariableOp_1?
$custom_model/StatefulPartitionedCallStatefulPartitionedCalltraining_inputcustom_model_2061custom_model_2063custom_model_2065custom_model_2067*
Tin	
2*
Tout	
2*
_collective_manager_ids
 *F
_output_shapes4
2:::::*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__forward_call_2512&
$custom_model/StatefulPartitionedCall?
$mean_squared_error/SquaredDifferenceSquaredDifference-custom_model/StatefulPartitionedCall:output:0training_target*
T0*
_output_shapes

:2&
$mean_squared_error/SquaredDifference?
)mean_squared_error/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)mean_squared_error/Mean/reduction_indices?
mean_squared_error/MeanMean(mean_squared_error/SquaredDifference:z:02mean_squared_error/Mean/reduction_indices:output:0*
T0*
_output_shapes
:2
mean_squared_error/Mean?
&mean_squared_error/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&mean_squared_error/weighted_loss/Const?
$mean_squared_error/weighted_loss/MulMul mean_squared_error/Mean:output:0/mean_squared_error/weighted_loss/Const:output:0*
T0*
_output_shapes
:2&
$mean_squared_error/weighted_loss/Mul?
(mean_squared_error/weighted_loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(mean_squared_error/weighted_loss/Const_1?
$mean_squared_error/weighted_loss/SumSum(mean_squared_error/weighted_loss/Mul:z:01mean_squared_error/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: 2&
$mean_squared_error/weighted_loss/Sum?
-mean_squared_error/weighted_loss/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :2/
-mean_squared_error/weighted_loss/num_elements?
2mean_squared_error/weighted_loss/num_elements/CastCast6mean_squared_error/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: 24
2mean_squared_error/weighted_loss/num_elements/Cast?
%mean_squared_error/weighted_loss/RankConst*
_output_shapes
: *
dtype0*
value	B : 2'
%mean_squared_error/weighted_loss/Rank?
,mean_squared_error/weighted_loss/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2.
,mean_squared_error/weighted_loss/range/start?
,mean_squared_error/weighted_loss/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2.
,mean_squared_error/weighted_loss/range/delta?
&mean_squared_error/weighted_loss/rangeRange5mean_squared_error/weighted_loss/range/start:output:0.mean_squared_error/weighted_loss/Rank:output:05mean_squared_error/weighted_loss/range/delta:output:0*
_output_shapes
: 2(
&mean_squared_error/weighted_loss/range?
&mean_squared_error/weighted_loss/Sum_1Sum-mean_squared_error/weighted_loss/Sum:output:0/mean_squared_error/weighted_loss/range:output:0*
T0*
_output_shapes
: 2(
&mean_squared_error/weighted_loss/Sum_1?
&mean_squared_error/weighted_loss/valueDivNoNan/mean_squared_error/weighted_loss/Sum_1:output:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 2(
&mean_squared_error/weighted_loss/value_
ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
CastCaststrided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Casth
MulMul*mean_squared_error/weighted_loss/value:z:0Cast:y:0*
T0*
_output_shapes
: 2
MulN
RankConst*
_output_shapes
: *
dtype0*
value	B : 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltal
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: 2
rangeK
SumSumMul:z:0range:output:0*
T0*
_output_shapes
: 2
Sum?
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype02
AssignAddVariableOpR
Rank_1Const*
_output_shapes
: *
dtype0*
value	B : 2
Rank_1`
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range_1/start`
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_1/deltav
range_1Rangerange_1/start:output:0Rank_1:output:0range_1/delta:output:0*
_output_shapes
: 2	
range_1R
Sum_1SumCast:y:0range_1:output:0*
T0*
_output_shapes
: 2
Sum_1?
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceSum_1:output:0^AssignAddVariableOp*
_output_shapes
 *
dtype02
AssignAddVariableOp_1m
SGD/gradients/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
SGD/gradients/ones?
:gradient_tape/mean_squared_error/weighted_loss/value/ShapeConst*
_output_shapes
: *
dtype0*
valueB 2<
:gradient_tape/mean_squared_error/weighted_loss/value/Shape?
<gradient_tape/mean_squared_error/weighted_loss/value/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2>
<gradient_tape/mean_squared_error/weighted_loss/value/Shape_1?
Jgradient_tape/mean_squared_error/weighted_loss/value/BroadcastGradientArgsBroadcastGradientArgsCgradient_tape/mean_squared_error/weighted_loss/value/Shape:output:0Egradient_tape/mean_squared_error/weighted_loss/value/Shape_1:output:0*2
_output_shapes 
:?????????:?????????2L
Jgradient_tape/mean_squared_error/weighted_loss/value/BroadcastGradientArgs?
?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nanDivNoNanSGD/gradients/ones:output:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 2A
?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan?
8gradient_tape/mean_squared_error/weighted_loss/value/SumSumCgradient_tape/mean_squared_error/weighted_loss/value/div_no_nan:z:0Ogradient_tape/mean_squared_error/weighted_loss/value/BroadcastGradientArgs:r0:0*
T0*
_output_shapes
: 2:
8gradient_tape/mean_squared_error/weighted_loss/value/Sum?
<gradient_tape/mean_squared_error/weighted_loss/value/ReshapeReshapeAgradient_tape/mean_squared_error/weighted_loss/value/Sum:output:0Cgradient_tape/mean_squared_error/weighted_loss/value/Shape:output:0*
T0*
_output_shapes
: 2>
<gradient_tape/mean_squared_error/weighted_loss/value/Reshape?
8gradient_tape/mean_squared_error/weighted_loss/value/NegNeg/mean_squared_error/weighted_loss/Sum_1:output:0*
T0*
_output_shapes
: 2:
8gradient_tape/mean_squared_error/weighted_loss/value/Neg?
Agradient_tape/mean_squared_error/weighted_loss/value/div_no_nan_1DivNoNan<gradient_tape/mean_squared_error/weighted_loss/value/Neg:y:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 2C
Agradient_tape/mean_squared_error/weighted_loss/value/div_no_nan_1?
Agradient_tape/mean_squared_error/weighted_loss/value/div_no_nan_2DivNoNanEgradient_tape/mean_squared_error/weighted_loss/value/div_no_nan_1:z:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 2C
Agradient_tape/mean_squared_error/weighted_loss/value/div_no_nan_2?
8gradient_tape/mean_squared_error/weighted_loss/value/mulMulSGD/gradients/ones:output:0Egradient_tape/mean_squared_error/weighted_loss/value/div_no_nan_2:z:0*
T0*
_output_shapes
: 2:
8gradient_tape/mean_squared_error/weighted_loss/value/mul?
:gradient_tape/mean_squared_error/weighted_loss/value/Sum_1Sum<gradient_tape/mean_squared_error/weighted_loss/value/mul:z:0Ogradient_tape/mean_squared_error/weighted_loss/value/BroadcastGradientArgs:r1:0*
T0*
_output_shapes
: 2<
:gradient_tape/mean_squared_error/weighted_loss/value/Sum_1?
>gradient_tape/mean_squared_error/weighted_loss/value/Reshape_1ReshapeCgradient_tape/mean_squared_error/weighted_loss/value/Sum_1:output:0Egradient_tape/mean_squared_error/weighted_loss/value/Shape_1:output:0*
T0*
_output_shapes
: 2@
>gradient_tape/mean_squared_error/weighted_loss/value/Reshape_1?
<gradient_tape/mean_squared_error/weighted_loss/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2>
<gradient_tape/mean_squared_error/weighted_loss/Reshape/shape?
>gradient_tape/mean_squared_error/weighted_loss/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2@
>gradient_tape/mean_squared_error/weighted_loss/Reshape/shape_1?
6gradient_tape/mean_squared_error/weighted_loss/ReshapeReshapeEgradient_tape/mean_squared_error/weighted_loss/value/Reshape:output:0Ggradient_tape/mean_squared_error/weighted_loss/Reshape/shape_1:output:0*
T0*
_output_shapes
: 28
6gradient_tape/mean_squared_error/weighted_loss/Reshape?
4gradient_tape/mean_squared_error/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB 26
4gradient_tape/mean_squared_error/weighted_loss/Const?
3gradient_tape/mean_squared_error/weighted_loss/TileTile?gradient_tape/mean_squared_error/weighted_loss/Reshape:output:0=gradient_tape/mean_squared_error/weighted_loss/Const:output:0*
T0*
_output_shapes
: 25
3gradient_tape/mean_squared_error/weighted_loss/Tile?
>gradient_tape/mean_squared_error/weighted_loss/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:2@
>gradient_tape/mean_squared_error/weighted_loss/Reshape_1/shape?
8gradient_tape/mean_squared_error/weighted_loss/Reshape_1Reshape<gradient_tape/mean_squared_error/weighted_loss/Tile:output:0Ggradient_tape/mean_squared_error/weighted_loss/Reshape_1/shape:output:0*
T0*
_output_shapes
:2:
8gradient_tape/mean_squared_error/weighted_loss/Reshape_1?
6gradient_tape/mean_squared_error/weighted_loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB:28
6gradient_tape/mean_squared_error/weighted_loss/Const_1?
5gradient_tape/mean_squared_error/weighted_loss/Tile_1TileAgradient_tape/mean_squared_error/weighted_loss/Reshape_1:output:0?gradient_tape/mean_squared_error/weighted_loss/Const_1:output:0*
T0*
_output_shapes
:27
5gradient_tape/mean_squared_error/weighted_loss/Tile_1?
2gradient_tape/mean_squared_error/weighted_loss/MulMul>gradient_tape/mean_squared_error/weighted_loss/Tile_1:output:0/mean_squared_error/weighted_loss/Const:output:0*
T0*
_output_shapes
:24
2gradient_tape/mean_squared_error/weighted_loss/Mul?
*gradient_tape/mean_squared_error/Maximum/xConst*
_output_shapes
:*
dtype0*
valueB"      2,
*gradient_tape/mean_squared_error/Maximum/x?
*gradient_tape/mean_squared_error/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :2,
*gradient_tape/mean_squared_error/Maximum/y?
(gradient_tape/mean_squared_error/MaximumMaximum3gradient_tape/mean_squared_error/Maximum/x:output:03gradient_tape/mean_squared_error/Maximum/y:output:0*
T0*
_output_shapes
:2*
(gradient_tape/mean_squared_error/Maximum?
+gradient_tape/mean_squared_error/floordiv/xConst*
_output_shapes
:*
dtype0*
valueB"      2-
+gradient_tape/mean_squared_error/floordiv/x?
)gradient_tape/mean_squared_error/floordivFloorDiv4gradient_tape/mean_squared_error/floordiv/x:output:0,gradient_tape/mean_squared_error/Maximum:z:0*
T0*
_output_shapes
:2+
)gradient_tape/mean_squared_error/floordiv?
.gradient_tape/mean_squared_error/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      20
.gradient_tape/mean_squared_error/Reshape/shape?
(gradient_tape/mean_squared_error/ReshapeReshape6gradient_tape/mean_squared_error/weighted_loss/Mul:z:07gradient_tape/mean_squared_error/Reshape/shape:output:0*
T0*
_output_shapes

:2*
(gradient_tape/mean_squared_error/Reshape?
/gradient_tape/mean_squared_error/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      21
/gradient_tape/mean_squared_error/Tile/multiples?
%gradient_tape/mean_squared_error/TileTile1gradient_tape/mean_squared_error/Reshape:output:08gradient_tape/mean_squared_error/Tile/multiples:output:0*
T0*
_output_shapes

:2'
%gradient_tape/mean_squared_error/Tile?
&gradient_tape/mean_squared_error/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&gradient_tape/mean_squared_error/Const?
(gradient_tape/mean_squared_error/truedivRealDiv.gradient_tape/mean_squared_error/Tile:output:0/gradient_tape/mean_squared_error/Const:output:0*
T0*
_output_shapes

:2*
(gradient_tape/mean_squared_error/truediv?
'gradient_tape/mean_squared_error/scalarConst)^gradient_tape/mean_squared_error/truediv*
_output_shapes
: *
dtype0*
valueB
 *   @2)
'gradient_tape/mean_squared_error/scalar?
$gradient_tape/mean_squared_error/MulMul0gradient_tape/mean_squared_error/scalar:output:0,gradient_tape/mean_squared_error/truediv:z:0*
T0*
_output_shapes

:2&
$gradient_tape/mean_squared_error/Mul?
$gradient_tape/mean_squared_error/subSub-custom_model/StatefulPartitionedCall:output:0training_target)^gradient_tape/mean_squared_error/truediv*
T0*
_output_shapes

:2&
$gradient_tape/mean_squared_error/sub?
&gradient_tape/mean_squared_error/mul_1Mul(gradient_tape/mean_squared_error/Mul:z:0(gradient_tape/mean_squared_error/sub:z:0*
T0*
_output_shapes

:2(
&gradient_tape/mean_squared_error/mul_1?
$gradient_tape/mean_squared_error/NegNeg*gradient_tape/mean_squared_error/mul_1:z:0*
T0*
_output_shapes

:2&
$gradient_tape/mean_squared_error/Neg?
SGD/gradients/PartitionedCallPartitionedCall*gradient_tape/mean_squared_error/mul_1:z:0-custom_model/StatefulPartitionedCall:output:1-custom_model/StatefulPartitionedCall:output:2-custom_model/StatefulPartitionedCall:output:3-custom_model/StatefulPartitionedCall:output:4*
Tin	
2*
Tout	
2*
_collective_manager_ids
 *>
_output_shapes,
*:::::* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference___backward_call_232_2522
SGD/gradients/PartitionedCall?
SGD/Cast/ReadVariableOpReadVariableOp sgd_cast_readvariableop_resource*
_output_shapes
: *
dtype02
SGD/Cast/ReadVariableOp?
SGD/IdentityIdentitySGD/Cast/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: 2
SGD/Identity?
SGD/Cast_1/ReadVariableOpReadVariableOp"sgd_cast_1_readvariableop_resource*
_output_shapes
: *
dtype02
SGD/Cast_1/ReadVariableOp?
SGD/Identity_1Identity!SGD/Cast_1/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: 2
SGD/Identity_1?
+SGD/SGD/update/ResourceApplyGradientDescentResourceApplyGradientDescentcustom_model_2061SGD/Identity:output:0&SGD/gradients/PartitionedCall:output:1%^custom_model/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*$
_class
loc:@custom_model/2061*
_output_shapes
 *
use_locking(2-
+SGD/SGD/update/ResourceApplyGradientDescent?
-SGD/SGD/update_1/ResourceApplyGradientDescentResourceApplyGradientDescentcustom_model_2063SGD/Identity:output:0&SGD/gradients/PartitionedCall:output:2%^custom_model/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*$
_class
loc:@custom_model/2063*
_output_shapes
 *
use_locking(2/
-SGD/SGD/update_1/ResourceApplyGradientDescent?
-SGD/SGD/update_2/ResourceApplyGradientDescentResourceApplyGradientDescentcustom_model_2065SGD/Identity:output:0&SGD/gradients/PartitionedCall:output:3%^custom_model/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*$
_class
loc:@custom_model/2065*
_output_shapes
 *
use_locking(2/
-SGD/SGD/update_2/ResourceApplyGradientDescent?
-SGD/SGD/update_3/ResourceApplyGradientDescentResourceApplyGradientDescentcustom_model_2067SGD/Identity:output:0&SGD/gradients/PartitionedCall:output:4%^custom_model/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*$
_class
loc:@custom_model/2067*
_output_shapes
 *
use_locking(2/
-SGD/SGD/update_3/ResourceApplyGradientDescent?
SGD/SGD/group_depsNoOp,^SGD/SGD/update/ResourceApplyGradientDescent.^SGD/SGD/update_1/ResourceApplyGradientDescent.^SGD/SGD/update_2/ResourceApplyGradientDescent.^SGD/SGD/update_3/ResourceApplyGradientDescent",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2
SGD/SGD/group_depsu
SGD/SGD/ConstConst^SGD/SGD/group_deps*
_output_shapes
: *
dtype0	*
value	B	 R2
SGD/SGD/Const?
SGD/SGD/AssignAddVariableOpAssignAddVariableOp$sgd_sgd_assignaddvariableop_resourceSGD/SGD/Const:output:0*
_output_shapes
 *
dtype0	2
SGD/SGD/AssignAddVariableOpz
subSub-custom_model/StatefulPartitionedCall:output:0training_target*
T0*
_output_shapes

:2
subC
AbsAbssub:z:0*
T0*
_output_shapes

:2
Abs{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Mean/reduction_indicesc
MeanMeanAbs:y:0Mean/reduction_indices:output:0*
T0*
_output_shapes
:2
MeanX
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstU
Sum_2SumMean:output:0Const:output:0*
T0*
_output_shapes
: 2
Sum_2?
AssignAddVariableOp_2AssignAddVariableOpassignaddvariableop_2_resourceSum_2:output:0*
_output_shapes
 *
dtype02
AssignAddVariableOp_2N
SizeConst*
_output_shapes
: *
dtype0*
value	B :2
SizeW
Cast_1CastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1?
AssignAddVariableOp_3AssignAddVariableOpassignaddvariableop_3_resource
Cast_1:y:0^AssignAddVariableOp_2*
_output_shapes
 *
dtype02
AssignAddVariableOp_3?
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp*
_output_shapes
: *
dtype02
div_no_nan/ReadVariableOp?
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype02
div_no_nan/ReadVariableOp_1?

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: 2

div_no_nanQ
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: 2

Identity?
div_no_nan_1/ReadVariableOpReadVariableOpassignaddvariableop_2_resource^AssignAddVariableOp_2*
_output_shapes
: *
dtype02
div_no_nan_1/ReadVariableOp?
div_no_nan_1/ReadVariableOp_1ReadVariableOpassignaddvariableop_3_resource^AssignAddVariableOp_3*
_output_shapes
: *
dtype02
div_no_nan_1/ReadVariableOp_1?
div_no_nan_1DivNoNan#div_no_nan_1/ReadVariableOp:value:0%div_no_nan_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
: 2
div_no_nan_1W

Identity_1Identitydiv_no_nan_1:z:0*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityIdentity:output:0^AssignAddVariableOp^AssignAddVariableOp_1^AssignAddVariableOp_2^AssignAddVariableOp_3^SGD/Cast/ReadVariableOp^SGD/Cast_1/ReadVariableOp^SGD/SGD/AssignAddVariableOp,^SGD/SGD/update/ResourceApplyGradientDescent.^SGD/SGD/update_1/ResourceApplyGradientDescent.^SGD/SGD/update_2/ResourceApplyGradientDescent.^SGD/SGD/update_3/ResourceApplyGradientDescent%^custom_model/StatefulPartitionedCall^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1^div_no_nan_1/ReadVariableOp^div_no_nan_1/ReadVariableOp_1*
T0*
_output_shapes
: 2

Identity_2"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*::: : : : : : : : : : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12.
AssignAddVariableOp_2AssignAddVariableOp_22.
AssignAddVariableOp_3AssignAddVariableOp_322
SGD/Cast/ReadVariableOpSGD/Cast/ReadVariableOp26
SGD/Cast_1/ReadVariableOpSGD/Cast_1/ReadVariableOp2:
SGD/SGD/AssignAddVariableOpSGD/SGD/AssignAddVariableOp2Z
+SGD/SGD/update/ResourceApplyGradientDescent+SGD/SGD/update/ResourceApplyGradientDescent2^
-SGD/SGD/update_1/ResourceApplyGradientDescent-SGD/SGD/update_1/ResourceApplyGradientDescent2^
-SGD/SGD/update_2/ResourceApplyGradientDescent-SGD/SGD/update_2/ResourceApplyGradientDescent2^
-SGD/SGD/update_3/ResourceApplyGradientDescent-SGD/SGD/update_3/ResourceApplyGradientDescent2L
$custom_model/StatefulPartitionedCall$custom_model/StatefulPartitionedCall26
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_12:
div_no_nan_1/ReadVariableOpdiv_no_nan_1/ReadVariableOp2>
div_no_nan_1/ReadVariableOp_1div_no_nan_1/ReadVariableOp_1:N J

_output_shapes

:
(
_user_specified_nametraining_input:OK

_output_shapes

:
)
_user_specified_nametraining_target
?
?
__forward_call_251
inputs_08
&test_in_matmul_readvariableop_resource:5
'test_in_biasadd_readvariableop_resource:9
'test_out_matmul_readvariableop_resource:6
(test_out_biasadd_readvariableop_resource:
identity"
test_out_matmul_readvariableop
test_in_biasadd!
test_in_matmul_readvariableop

inputs??test_in/BiasAdd/ReadVariableOp?test_in/MatMul/ReadVariableOp?test_out/BiasAdd/ReadVariableOp?test_out/MatMul/ReadVariableOp?
test_in/MatMul/ReadVariableOpReadVariableOp&test_in_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
test_in/MatMul/ReadVariableOp?
test_in/MatMulMatMulinputs_0%test_in/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_in/MatMul?
test_in/BiasAdd/ReadVariableOpReadVariableOp'test_in_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
test_in/BiasAdd/ReadVariableOp?
test_in/BiasAddBiasAddtest_in/MatMul:product:0&test_in/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_in/BiasAdd?
test_out/MatMul/ReadVariableOpReadVariableOp'test_out_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
test_out/MatMul/ReadVariableOp?
test_out/MatMulMatMultest_in/BiasAdd:output:0&test_out/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_out/MatMul?
test_out/BiasAdd/ReadVariableOpReadVariableOp(test_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
test_out/BiasAdd/ReadVariableOp?
test_out/BiasAddBiasAddtest_out/MatMul:product:0'test_out/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_out/BiasAdd?
IdentityIdentitytest_out/BiasAdd:output:0^test_in/BiasAdd/ReadVariableOp^test_in/MatMul/ReadVariableOp ^test_out/BiasAdd/ReadVariableOp^test_out/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0"
inputsinputs_0"+
test_in_biasaddtest_in/BiasAdd:output:0"F
test_in_matmul_readvariableop%test_in/MatMul/ReadVariableOp:value:0"H
test_out_matmul_readvariableop&test_out/MatMul/ReadVariableOp:value:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : : : *?
backward_function_name%#__inference___backward_call_232_2522@
test_in/BiasAdd/ReadVariableOptest_in/BiasAdd/ReadVariableOp2>
test_in/MatMul/ReadVariableOptest_in/MatMul/ReadVariableOp2B
test_out/BiasAdd/ReadVariableOptest_out/BiasAdd/ReadVariableOp2@
test_out/MatMul/ReadVariableOptest_out/MatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
??
?
__inference_training_2531
training_input
training_target#
custom_model_2387:
custom_model_2389:#
custom_model_2391:
custom_model_2393:&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: *
 sgd_cast_readvariableop_resource: ,
"sgd_cast_1_readvariableop_resource: .
$sgd_sgd_assignaddvariableop_resource:	 (
assignaddvariableop_2_resource: (
assignaddvariableop_3_resource: 

identity_2??AssignAddVariableOp?AssignAddVariableOp_1?AssignAddVariableOp_2?AssignAddVariableOp_3?SGD/Cast/ReadVariableOp?SGD/Cast_1/ReadVariableOp?SGD/SGD/AssignAddVariableOp?+SGD/SGD/update/ResourceApplyGradientDescent?-SGD/SGD/update_1/ResourceApplyGradientDescent?-SGD/SGD/update_2/ResourceApplyGradientDescent?-SGD/SGD/update_3/ResourceApplyGradientDescent?$custom_model/StatefulPartitionedCall?div_no_nan/ReadVariableOp?div_no_nan/ReadVariableOp_1?div_no_nan_1/ReadVariableOp?div_no_nan_1/ReadVariableOp_1?
$custom_model/StatefulPartitionedCallStatefulPartitionedCalltraining_inputcustom_model_2387custom_model_2389custom_model_2391custom_model_2393*
Tin	
2*
Tout	
2*
_collective_manager_ids
 *F
_output_shapes4
2:::::*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__forward_call_24892&
$custom_model/StatefulPartitionedCall?
$mean_squared_error/SquaredDifferenceSquaredDifference-custom_model/StatefulPartitionedCall:output:0training_target*
T0*
_output_shapes

:2&
$mean_squared_error/SquaredDifference?
)mean_squared_error/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)mean_squared_error/Mean/reduction_indices?
mean_squared_error/MeanMean(mean_squared_error/SquaredDifference:z:02mean_squared_error/Mean/reduction_indices:output:0*
T0*
_output_shapes
:2
mean_squared_error/Mean?
&mean_squared_error/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&mean_squared_error/weighted_loss/Const?
$mean_squared_error/weighted_loss/MulMul mean_squared_error/Mean:output:0/mean_squared_error/weighted_loss/Const:output:0*
T0*
_output_shapes
:2&
$mean_squared_error/weighted_loss/Mul?
(mean_squared_error/weighted_loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(mean_squared_error/weighted_loss/Const_1?
$mean_squared_error/weighted_loss/SumSum(mean_squared_error/weighted_loss/Mul:z:01mean_squared_error/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: 2&
$mean_squared_error/weighted_loss/Sum?
-mean_squared_error/weighted_loss/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :2/
-mean_squared_error/weighted_loss/num_elements?
2mean_squared_error/weighted_loss/num_elements/CastCast6mean_squared_error/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: 24
2mean_squared_error/weighted_loss/num_elements/Cast?
%mean_squared_error/weighted_loss/RankConst*
_output_shapes
: *
dtype0*
value	B : 2'
%mean_squared_error/weighted_loss/Rank?
,mean_squared_error/weighted_loss/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2.
,mean_squared_error/weighted_loss/range/start?
,mean_squared_error/weighted_loss/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2.
,mean_squared_error/weighted_loss/range/delta?
&mean_squared_error/weighted_loss/rangeRange5mean_squared_error/weighted_loss/range/start:output:0.mean_squared_error/weighted_loss/Rank:output:05mean_squared_error/weighted_loss/range/delta:output:0*
_output_shapes
: 2(
&mean_squared_error/weighted_loss/range?
&mean_squared_error/weighted_loss/Sum_1Sum-mean_squared_error/weighted_loss/Sum:output:0/mean_squared_error/weighted_loss/range:output:0*
T0*
_output_shapes
: 2(
&mean_squared_error/weighted_loss/Sum_1?
&mean_squared_error/weighted_loss/valueDivNoNan/mean_squared_error/weighted_loss/Sum_1:output:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 2(
&mean_squared_error/weighted_loss/value_
ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
CastCaststrided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Casth
MulMul*mean_squared_error/weighted_loss/value:z:0Cast:y:0*
T0*
_output_shapes
: 2
MulN
RankConst*
_output_shapes
: *
dtype0*
value	B : 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltal
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: 2
rangeK
SumSumMul:z:0range:output:0*
T0*
_output_shapes
: 2
Sum?
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype02
AssignAddVariableOpR
Rank_1Const*
_output_shapes
: *
dtype0*
value	B : 2
Rank_1`
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range_1/start`
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_1/deltav
range_1Rangerange_1/start:output:0Rank_1:output:0range_1/delta:output:0*
_output_shapes
: 2	
range_1R
Sum_1SumCast:y:0range_1:output:0*
T0*
_output_shapes
: 2
Sum_1?
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceSum_1:output:0^AssignAddVariableOp*
_output_shapes
 *
dtype02
AssignAddVariableOp_1m
SGD/gradients/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
SGD/gradients/ones?
:gradient_tape/mean_squared_error/weighted_loss/value/ShapeConst*
_output_shapes
: *
dtype0*
valueB 2<
:gradient_tape/mean_squared_error/weighted_loss/value/Shape?
<gradient_tape/mean_squared_error/weighted_loss/value/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2>
<gradient_tape/mean_squared_error/weighted_loss/value/Shape_1?
Jgradient_tape/mean_squared_error/weighted_loss/value/BroadcastGradientArgsBroadcastGradientArgsCgradient_tape/mean_squared_error/weighted_loss/value/Shape:output:0Egradient_tape/mean_squared_error/weighted_loss/value/Shape_1:output:0*2
_output_shapes 
:?????????:?????????2L
Jgradient_tape/mean_squared_error/weighted_loss/value/BroadcastGradientArgs?
?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nanDivNoNanSGD/gradients/ones:output:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 2A
?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan?
8gradient_tape/mean_squared_error/weighted_loss/value/SumSumCgradient_tape/mean_squared_error/weighted_loss/value/div_no_nan:z:0Ogradient_tape/mean_squared_error/weighted_loss/value/BroadcastGradientArgs:r0:0*
T0*
_output_shapes
: 2:
8gradient_tape/mean_squared_error/weighted_loss/value/Sum?
<gradient_tape/mean_squared_error/weighted_loss/value/ReshapeReshapeAgradient_tape/mean_squared_error/weighted_loss/value/Sum:output:0Cgradient_tape/mean_squared_error/weighted_loss/value/Shape:output:0*
T0*
_output_shapes
: 2>
<gradient_tape/mean_squared_error/weighted_loss/value/Reshape?
8gradient_tape/mean_squared_error/weighted_loss/value/NegNeg/mean_squared_error/weighted_loss/Sum_1:output:0*
T0*
_output_shapes
: 2:
8gradient_tape/mean_squared_error/weighted_loss/value/Neg?
Agradient_tape/mean_squared_error/weighted_loss/value/div_no_nan_1DivNoNan<gradient_tape/mean_squared_error/weighted_loss/value/Neg:y:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 2C
Agradient_tape/mean_squared_error/weighted_loss/value/div_no_nan_1?
Agradient_tape/mean_squared_error/weighted_loss/value/div_no_nan_2DivNoNanEgradient_tape/mean_squared_error/weighted_loss/value/div_no_nan_1:z:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: 2C
Agradient_tape/mean_squared_error/weighted_loss/value/div_no_nan_2?
8gradient_tape/mean_squared_error/weighted_loss/value/mulMulSGD/gradients/ones:output:0Egradient_tape/mean_squared_error/weighted_loss/value/div_no_nan_2:z:0*
T0*
_output_shapes
: 2:
8gradient_tape/mean_squared_error/weighted_loss/value/mul?
:gradient_tape/mean_squared_error/weighted_loss/value/Sum_1Sum<gradient_tape/mean_squared_error/weighted_loss/value/mul:z:0Ogradient_tape/mean_squared_error/weighted_loss/value/BroadcastGradientArgs:r1:0*
T0*
_output_shapes
: 2<
:gradient_tape/mean_squared_error/weighted_loss/value/Sum_1?
>gradient_tape/mean_squared_error/weighted_loss/value/Reshape_1ReshapeCgradient_tape/mean_squared_error/weighted_loss/value/Sum_1:output:0Egradient_tape/mean_squared_error/weighted_loss/value/Shape_1:output:0*
T0*
_output_shapes
: 2@
>gradient_tape/mean_squared_error/weighted_loss/value/Reshape_1?
<gradient_tape/mean_squared_error/weighted_loss/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2>
<gradient_tape/mean_squared_error/weighted_loss/Reshape/shape?
>gradient_tape/mean_squared_error/weighted_loss/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2@
>gradient_tape/mean_squared_error/weighted_loss/Reshape/shape_1?
6gradient_tape/mean_squared_error/weighted_loss/ReshapeReshapeEgradient_tape/mean_squared_error/weighted_loss/value/Reshape:output:0Ggradient_tape/mean_squared_error/weighted_loss/Reshape/shape_1:output:0*
T0*
_output_shapes
: 28
6gradient_tape/mean_squared_error/weighted_loss/Reshape?
4gradient_tape/mean_squared_error/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB 26
4gradient_tape/mean_squared_error/weighted_loss/Const?
3gradient_tape/mean_squared_error/weighted_loss/TileTile?gradient_tape/mean_squared_error/weighted_loss/Reshape:output:0=gradient_tape/mean_squared_error/weighted_loss/Const:output:0*
T0*
_output_shapes
: 25
3gradient_tape/mean_squared_error/weighted_loss/Tile?
>gradient_tape/mean_squared_error/weighted_loss/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:2@
>gradient_tape/mean_squared_error/weighted_loss/Reshape_1/shape?
8gradient_tape/mean_squared_error/weighted_loss/Reshape_1Reshape<gradient_tape/mean_squared_error/weighted_loss/Tile:output:0Ggradient_tape/mean_squared_error/weighted_loss/Reshape_1/shape:output:0*
T0*
_output_shapes
:2:
8gradient_tape/mean_squared_error/weighted_loss/Reshape_1?
6gradient_tape/mean_squared_error/weighted_loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB:28
6gradient_tape/mean_squared_error/weighted_loss/Const_1?
5gradient_tape/mean_squared_error/weighted_loss/Tile_1TileAgradient_tape/mean_squared_error/weighted_loss/Reshape_1:output:0?gradient_tape/mean_squared_error/weighted_loss/Const_1:output:0*
T0*
_output_shapes
:27
5gradient_tape/mean_squared_error/weighted_loss/Tile_1?
2gradient_tape/mean_squared_error/weighted_loss/MulMul>gradient_tape/mean_squared_error/weighted_loss/Tile_1:output:0/mean_squared_error/weighted_loss/Const:output:0*
T0*
_output_shapes
:24
2gradient_tape/mean_squared_error/weighted_loss/Mul?
*gradient_tape/mean_squared_error/Maximum/xConst*
_output_shapes
:*
dtype0*
valueB"      2,
*gradient_tape/mean_squared_error/Maximum/x?
*gradient_tape/mean_squared_error/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :2,
*gradient_tape/mean_squared_error/Maximum/y?
(gradient_tape/mean_squared_error/MaximumMaximum3gradient_tape/mean_squared_error/Maximum/x:output:03gradient_tape/mean_squared_error/Maximum/y:output:0*
T0*
_output_shapes
:2*
(gradient_tape/mean_squared_error/Maximum?
+gradient_tape/mean_squared_error/floordiv/xConst*
_output_shapes
:*
dtype0*
valueB"      2-
+gradient_tape/mean_squared_error/floordiv/x?
)gradient_tape/mean_squared_error/floordivFloorDiv4gradient_tape/mean_squared_error/floordiv/x:output:0,gradient_tape/mean_squared_error/Maximum:z:0*
T0*
_output_shapes
:2+
)gradient_tape/mean_squared_error/floordiv?
.gradient_tape/mean_squared_error/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      20
.gradient_tape/mean_squared_error/Reshape/shape?
(gradient_tape/mean_squared_error/ReshapeReshape6gradient_tape/mean_squared_error/weighted_loss/Mul:z:07gradient_tape/mean_squared_error/Reshape/shape:output:0*
T0*
_output_shapes

:2*
(gradient_tape/mean_squared_error/Reshape?
/gradient_tape/mean_squared_error/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      21
/gradient_tape/mean_squared_error/Tile/multiples?
%gradient_tape/mean_squared_error/TileTile1gradient_tape/mean_squared_error/Reshape:output:08gradient_tape/mean_squared_error/Tile/multiples:output:0*
T0*
_output_shapes

:2'
%gradient_tape/mean_squared_error/Tile?
&gradient_tape/mean_squared_error/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&gradient_tape/mean_squared_error/Const?
(gradient_tape/mean_squared_error/truedivRealDiv.gradient_tape/mean_squared_error/Tile:output:0/gradient_tape/mean_squared_error/Const:output:0*
T0*
_output_shapes

:2*
(gradient_tape/mean_squared_error/truediv?
'gradient_tape/mean_squared_error/scalarConst)^gradient_tape/mean_squared_error/truediv*
_output_shapes
: *
dtype0*
valueB
 *   @2)
'gradient_tape/mean_squared_error/scalar?
$gradient_tape/mean_squared_error/MulMul0gradient_tape/mean_squared_error/scalar:output:0,gradient_tape/mean_squared_error/truediv:z:0*
T0*
_output_shapes

:2&
$gradient_tape/mean_squared_error/Mul?
$gradient_tape/mean_squared_error/subSub-custom_model/StatefulPartitionedCall:output:0training_target)^gradient_tape/mean_squared_error/truediv*
T0*
_output_shapes

:2&
$gradient_tape/mean_squared_error/sub?
&gradient_tape/mean_squared_error/mul_1Mul(gradient_tape/mean_squared_error/Mul:z:0(gradient_tape/mean_squared_error/sub:z:0*
T0*
_output_shapes

:2(
&gradient_tape/mean_squared_error/mul_1?
$gradient_tape/mean_squared_error/NegNeg*gradient_tape/mean_squared_error/mul_1:z:0*
T0*
_output_shapes

:2&
$gradient_tape/mean_squared_error/Neg?
SGD/gradients/PartitionedCallPartitionedCall*gradient_tape/mean_squared_error/mul_1:z:0-custom_model/StatefulPartitionedCall:output:1-custom_model/StatefulPartitionedCall:output:2-custom_model/StatefulPartitionedCall:output:3-custom_model/StatefulPartitionedCall:output:4*
Tin	
2*
Tout	
2*
_collective_manager_ids
 *>
_output_shapes,
*:::::* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference___backward_call_2470_24902
SGD/gradients/PartitionedCall?
SGD/Cast/ReadVariableOpReadVariableOp sgd_cast_readvariableop_resource*
_output_shapes
: *
dtype02
SGD/Cast/ReadVariableOp?
SGD/IdentityIdentitySGD/Cast/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: 2
SGD/Identity?
SGD/Cast_1/ReadVariableOpReadVariableOp"sgd_cast_1_readvariableop_resource*
_output_shapes
: *
dtype02
SGD/Cast_1/ReadVariableOp?
SGD/Identity_1Identity!SGD/Cast_1/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: 2
SGD/Identity_1?
+SGD/SGD/update/ResourceApplyGradientDescentResourceApplyGradientDescentcustom_model_2387SGD/Identity:output:0&SGD/gradients/PartitionedCall:output:1%^custom_model/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*$
_class
loc:@custom_model/2387*
_output_shapes
 *
use_locking(2-
+SGD/SGD/update/ResourceApplyGradientDescent?
-SGD/SGD/update_1/ResourceApplyGradientDescentResourceApplyGradientDescentcustom_model_2389SGD/Identity:output:0&SGD/gradients/PartitionedCall:output:2%^custom_model/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*$
_class
loc:@custom_model/2389*
_output_shapes
 *
use_locking(2/
-SGD/SGD/update_1/ResourceApplyGradientDescent?
-SGD/SGD/update_2/ResourceApplyGradientDescentResourceApplyGradientDescentcustom_model_2391SGD/Identity:output:0&SGD/gradients/PartitionedCall:output:3%^custom_model/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*$
_class
loc:@custom_model/2391*
_output_shapes
 *
use_locking(2/
-SGD/SGD/update_2/ResourceApplyGradientDescent?
-SGD/SGD/update_3/ResourceApplyGradientDescentResourceApplyGradientDescentcustom_model_2393SGD/Identity:output:0&SGD/gradients/PartitionedCall:output:4%^custom_model/StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:CPU:0*
T0*$
_class
loc:@custom_model/2393*
_output_shapes
 *
use_locking(2/
-SGD/SGD/update_3/ResourceApplyGradientDescent?
SGD/SGD/group_depsNoOp,^SGD/SGD/update/ResourceApplyGradientDescent.^SGD/SGD/update_1/ResourceApplyGradientDescent.^SGD/SGD/update_2/ResourceApplyGradientDescent.^SGD/SGD/update_3/ResourceApplyGradientDescent",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2
SGD/SGD/group_depsu
SGD/SGD/ConstConst^SGD/SGD/group_deps*
_output_shapes
: *
dtype0	*
value	B	 R2
SGD/SGD/Const?
SGD/SGD/AssignAddVariableOpAssignAddVariableOp$sgd_sgd_assignaddvariableop_resourceSGD/SGD/Const:output:0*
_output_shapes
 *
dtype0	2
SGD/SGD/AssignAddVariableOpz
subSub-custom_model/StatefulPartitionedCall:output:0training_target*
T0*
_output_shapes

:2
subC
AbsAbssub:z:0*
T0*
_output_shapes

:2
Abs{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Mean/reduction_indicesc
MeanMeanAbs:y:0Mean/reduction_indices:output:0*
T0*
_output_shapes
:2
MeanX
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstU
Sum_2SumMean:output:0Const:output:0*
T0*
_output_shapes
: 2
Sum_2?
AssignAddVariableOp_2AssignAddVariableOpassignaddvariableop_2_resourceSum_2:output:0*
_output_shapes
 *
dtype02
AssignAddVariableOp_2N
SizeConst*
_output_shapes
: *
dtype0*
value	B :2
SizeW
Cast_1CastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1?
AssignAddVariableOp_3AssignAddVariableOpassignaddvariableop_3_resource
Cast_1:y:0^AssignAddVariableOp_2*
_output_shapes
 *
dtype02
AssignAddVariableOp_3?
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp*
_output_shapes
: *
dtype02
div_no_nan/ReadVariableOp?
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype02
div_no_nan/ReadVariableOp_1?

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: 2

div_no_nanQ
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: 2

Identity?
div_no_nan_1/ReadVariableOpReadVariableOpassignaddvariableop_2_resource^AssignAddVariableOp_2*
_output_shapes
: *
dtype02
div_no_nan_1/ReadVariableOp?
div_no_nan_1/ReadVariableOp_1ReadVariableOpassignaddvariableop_3_resource^AssignAddVariableOp_3*
_output_shapes
: *
dtype02
div_no_nan_1/ReadVariableOp_1?
div_no_nan_1DivNoNan#div_no_nan_1/ReadVariableOp:value:0%div_no_nan_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
: 2
div_no_nan_1W

Identity_1Identitydiv_no_nan_1:z:0*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityIdentity:output:0^AssignAddVariableOp^AssignAddVariableOp_1^AssignAddVariableOp_2^AssignAddVariableOp_3^SGD/Cast/ReadVariableOp^SGD/Cast_1/ReadVariableOp^SGD/SGD/AssignAddVariableOp,^SGD/SGD/update/ResourceApplyGradientDescent.^SGD/SGD/update_1/ResourceApplyGradientDescent.^SGD/SGD/update_2/ResourceApplyGradientDescent.^SGD/SGD/update_3/ResourceApplyGradientDescent%^custom_model/StatefulPartitionedCall^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1^div_no_nan_1/ReadVariableOp^div_no_nan_1/ReadVariableOp_1*
T0*
_output_shapes
: 2

Identity_2"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*::: : : : : : : : : : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12.
AssignAddVariableOp_2AssignAddVariableOp_22.
AssignAddVariableOp_3AssignAddVariableOp_322
SGD/Cast/ReadVariableOpSGD/Cast/ReadVariableOp26
SGD/Cast_1/ReadVariableOpSGD/Cast_1/ReadVariableOp2:
SGD/SGD/AssignAddVariableOpSGD/SGD/AssignAddVariableOp2Z
+SGD/SGD/update/ResourceApplyGradientDescent+SGD/SGD/update/ResourceApplyGradientDescent2^
-SGD/SGD/update_1/ResourceApplyGradientDescent-SGD/SGD/update_1/ResourceApplyGradientDescent2^
-SGD/SGD/update_2/ResourceApplyGradientDescent-SGD/SGD/update_2/ResourceApplyGradientDescent2^
-SGD/SGD/update_3/ResourceApplyGradientDescent-SGD/SGD/update_3/ResourceApplyGradientDescent2L
$custom_model/StatefulPartitionedCall$custom_model/StatefulPartitionedCall26
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_12:
div_no_nan_1/ReadVariableOpdiv_no_nan_1/ReadVariableOp2>
div_no_nan_1/ReadVariableOp_1div_no_nan_1/ReadVariableOp_1:N J

_output_shapes

:
(
_user_specified_nametraining_input:OK

_output_shapes

:
)
_user_specified_nametraining_target
?
?
%__inference___backward_call_2470_2490
placeholderH
Dgradients_test_out_matmul_grad_matmul_test_out_matmul_readvariableop;
7gradients_test_out_matmul_grad_matmul_1_test_in_biasaddF
Bgradients_test_in_matmul_grad_matmul_test_in_matmul_readvariableop1
-gradients_test_in_matmul_grad_matmul_1_inputs
identity

identity_1

identity_2

identity_3

identity_4l
gradients/grad_ys_0Identityplaceholder*
T0*
_output_shapes

:2
gradients/grad_ys_0?
+gradients/test_out/BiasAdd_grad/BiasAddGradBiasAddGradgradients/grad_ys_0:output:0*
T0*
_output_shapes
:2-
+gradients/test_out/BiasAdd_grad/BiasAddGrad?
%gradients/test_out/MatMul_grad/MatMulMatMulgradients/grad_ys_0:output:0Dgradients_test_out_matmul_grad_matmul_test_out_matmul_readvariableop*
T0*
_output_shapes

:*
transpose_b(2'
%gradients/test_out/MatMul_grad/MatMul?
'gradients/test_out/MatMul_grad/MatMul_1MatMul7gradients_test_out_matmul_grad_matmul_1_test_in_biasaddgradients/grad_ys_0:output:0*
T0*
_output_shapes

:*
transpose_a(2)
'gradients/test_out/MatMul_grad/MatMul_1?
*gradients/test_in/BiasAdd_grad/BiasAddGradBiasAddGrad/gradients/test_out/MatMul_grad/MatMul:product:0*
T0*
_output_shapes
:2,
*gradients/test_in/BiasAdd_grad/BiasAddGrad?
$gradients/test_in/MatMul_grad/MatMulMatMul/gradients/test_out/MatMul_grad/MatMul:product:0Bgradients_test_in_matmul_grad_matmul_test_in_matmul_readvariableop*
T0*
_output_shapes

:*
transpose_b(2&
$gradients/test_in/MatMul_grad/MatMul?
&gradients/test_in/MatMul_grad/MatMul_1MatMul-gradients_test_in_matmul_grad_matmul_1_inputs/gradients/test_out/MatMul_grad/MatMul:product:0*
T0*
_output_shapes

:*
transpose_a(2(
&gradients/test_in/MatMul_grad/MatMul_1y
IdentityIdentity.gradients/test_in/MatMul_grad/MatMul:product:0*
T0*
_output_shapes

:2

Identity

Identity_1Identity0gradients/test_in/MatMul_grad/MatMul_1:product:0*
T0*
_output_shapes

:2

Identity_1~

Identity_2Identity3gradients/test_in/BiasAdd_grad/BiasAddGrad:output:0*
T0*
_output_shapes
:2

Identity_2?

Identity_3Identity1gradients/test_out/MatMul_grad/MatMul_1:product:0*
T0*
_output_shapes

:2

Identity_3

Identity_4Identity4gradients/test_out/BiasAdd_grad/BiasAddGrad:output:0*
T0*
_output_shapes
:2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:::::*.
forward_function_name__forward_call_2489:$  

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
?
__inference__wrapped_model_2260
input_1#
custom_model_2250:
custom_model_2252:#
custom_model_2254:
custom_model_2256:
identity??$custom_model/StatefulPartitionedCall?
$custom_model/StatefulPartitionedCallStatefulPartitionedCallinput_1custom_model_2250custom_model_2252custom_model_2254custom_model_2256*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_22492&
$custom_model/StatefulPartitionedCall?
IdentityIdentity-custom_model/StatefulPartitionedCall:output:0%^custom_model/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2L
$custom_model/StatefulPartitionedCall$custom_model/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
#__inference___backward_call_232_252
placeholderH
Dgradients_test_out_matmul_grad_matmul_test_out_matmul_readvariableop;
7gradients_test_out_matmul_grad_matmul_1_test_in_biasaddF
Bgradients_test_in_matmul_grad_matmul_test_in_matmul_readvariableop1
-gradients_test_in_matmul_grad_matmul_1_inputs
identity

identity_1

identity_2

identity_3

identity_4l
gradients/grad_ys_0Identityplaceholder*
T0*
_output_shapes

:2
gradients/grad_ys_0?
+gradients/test_out/BiasAdd_grad/BiasAddGradBiasAddGradgradients/grad_ys_0:output:0*
T0*
_output_shapes
:2-
+gradients/test_out/BiasAdd_grad/BiasAddGrad?
%gradients/test_out/MatMul_grad/MatMulMatMulgradients/grad_ys_0:output:0Dgradients_test_out_matmul_grad_matmul_test_out_matmul_readvariableop*
T0*
_output_shapes

:*
transpose_b(2'
%gradients/test_out/MatMul_grad/MatMul?
'gradients/test_out/MatMul_grad/MatMul_1MatMul7gradients_test_out_matmul_grad_matmul_1_test_in_biasaddgradients/grad_ys_0:output:0*
T0*
_output_shapes

:*
transpose_a(2)
'gradients/test_out/MatMul_grad/MatMul_1?
*gradients/test_in/BiasAdd_grad/BiasAddGradBiasAddGrad/gradients/test_out/MatMul_grad/MatMul:product:0*
T0*
_output_shapes
:2,
*gradients/test_in/BiasAdd_grad/BiasAddGrad?
$gradients/test_in/MatMul_grad/MatMulMatMul/gradients/test_out/MatMul_grad/MatMul:product:0Bgradients_test_in_matmul_grad_matmul_test_in_matmul_readvariableop*
T0*
_output_shapes

:*
transpose_b(2&
$gradients/test_in/MatMul_grad/MatMul?
&gradients/test_in/MatMul_grad/MatMul_1MatMul-gradients_test_in_matmul_grad_matmul_1_inputs/gradients/test_out/MatMul_grad/MatMul:product:0*
T0*
_output_shapes

:*
transpose_a(2(
&gradients/test_in/MatMul_grad/MatMul_1y
IdentityIdentity.gradients/test_in/MatMul_grad/MatMul:product:0*
T0*
_output_shapes

:2

Identity

Identity_1Identity0gradients/test_in/MatMul_grad/MatMul_1:product:0*
T0*
_output_shapes

:2

Identity_1~

Identity_2Identity3gradients/test_in/BiasAdd_grad/BiasAddGrad:output:0*
T0*
_output_shapes
:2

Identity_2?

Identity_3Identity1gradients/test_out/MatMul_grad/MatMul_1:product:0*
T0*
_output_shapes

:2

Identity_3

Identity_4Identity4gradients/test_out/BiasAdd_grad/BiasAddGrad:output:0*
T0*
_output_shapes
:2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:::::*-
forward_function_name__forward_call_251:$  

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
?
+__inference_custom_model_layer_call_fn_2311
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_custom_model_layer_call_and_return_conditional_losses_22972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
A__inference_test_in_layer_call_and_return_conditional_losses_2274

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_2230

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *
fR
__inference_call_20562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs
?
?
__inference_call_2386

inputs8
&test_in_matmul_readvariableop_resource:5
'test_in_biasadd_readvariableop_resource:9
'test_out_matmul_readvariableop_resource:6
(test_out_biasadd_readvariableop_resource:
identity??test_in/BiasAdd/ReadVariableOp?test_in/MatMul/ReadVariableOp?test_out/BiasAdd/ReadVariableOp?test_out/MatMul/ReadVariableOp?
test_in/MatMul/ReadVariableOpReadVariableOp&test_in_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
test_in/MatMul/ReadVariableOp?
test_in/MatMulMatMulinputs%test_in/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_in/MatMul?
test_in/BiasAdd/ReadVariableOpReadVariableOp'test_in_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
test_in/BiasAdd/ReadVariableOp?
test_in/BiasAddBiasAddtest_in/MatMul:product:0&test_in/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_in/BiasAdd?
test_out/MatMul/ReadVariableOpReadVariableOp'test_out_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
test_out/MatMul/ReadVariableOp?
test_out/MatMulMatMultest_in/BiasAdd:output:0&test_out/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_out/MatMul?
test_out/BiasAdd/ReadVariableOpReadVariableOp(test_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
test_out/BiasAdd/ReadVariableOp?
test_out/BiasAddBiasAddtest_out/MatMul:product:0'test_out/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
test_out/BiasAdd?
IdentityIdentitytest_out/BiasAdd:output:0^test_in/BiasAdd/ReadVariableOp^test_in/MatMul/ReadVariableOp ^test_out/BiasAdd/ReadVariableOp^test_out/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : : : 2@
test_in/BiasAdd/ReadVariableOptest_in/BiasAdd/ReadVariableOp2>
test_in/MatMul/ReadVariableOptest_in/MatMul/ReadVariableOp2B
test_out/BiasAdd/ReadVariableOptest_out/BiasAdd/ReadVariableOp2@
test_out/MatMul/ReadVariableOptest_out/MatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
predx
%
inputs
pred_inputs:03
output_0'
StatefulPartitionedCall:0tensorflow/serving/predict*?
train?
6
training_input$
train_training_input:0
8
training_target%
train_training_target:0-
output_0!
StatefulPartitionedCall_1:0 tensorflow/serving/predict:?T
?	
dense_1
dense_2
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
3__call__
*4&call_and_return_all_conditional_losses
5_default_save_signature
6call
7training"?
_tf_keras_model?{"name": "custom_model", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "custom_model", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 2]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "custom_model"}, "training_config": {"loss": "mse", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}, "shared_object_id": 0}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.009999999776482582, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
?	

	kernel

bias
regularization_losses
trainable_variables
	variables
	keras_api
8__call__
*9&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "test_in", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "test_in", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}, "shared_object_id": 4}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 2]}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
:__call__
*;&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "test_out", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "test_out", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}, "shared_object_id": 8}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 2]}}
I
iter
	decay
learning_rate
momentum"
	optimizer
 "
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
?
layer_metrics
non_trainable_variables
regularization_losses
layer_regularization_losses
trainable_variables

layers
metrics
	variables
3__call__
5_default_save_signature
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
,
	<train
=pred"
signature_map
 :2test_in/kernel
:2test_in/bias
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
?
layer_metrics
non_trainable_variables
regularization_losses
 layer_regularization_losses
trainable_variables

!layers
"metrics
	variables
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
!:2test_out/kernel
:2test_out/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
#layer_metrics
$non_trainable_variables
regularization_losses
%layer_regularization_losses
trainable_variables

&layers
'metrics
	variables
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	*total
	+count
,	variables
-	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 9}
?
	.total
	/count
0
_fn_kwargs
1	variables
2	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}, "shared_object_id": 0}
:  (2total
:  (2count
.
*0
+1"
trackable_list_wrapper
-
,	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
.0
/1"
trackable_list_wrapper
-
1	variables"
_generic_user_object
?2?
+__inference_custom_model_layer_call_fn_2311?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
F__inference_custom_model_layer_call_and_return_conditional_losses_2297?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
__inference__wrapped_model_2260?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
__inference_call_2351
__inference_call_2367?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_training_2531?
???
FullArgSpec!
args?
jself
j
train_data
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_test_in_layer_call_fn_2540?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_test_in_layer_call_and_return_conditional_losses_2550?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_test_out_layer_call_fn_2559?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_test_out_layer_call_and_return_conditional_losses_2569?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_2215training_inputtraining_target"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_2230inputs"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference__wrapped_model_2260m	
0?-
&?#
!?
input_1?????????
? "3?0
.
output_1"?
output_1?????????X
__inference_call_2351?	
&?#
?
?
inputs
? "?j
__inference_call_2367Q	
/?,
%?"
 ?
inputs?????????
? "???????????
F__inference_custom_model_layer_call_and_return_conditional_losses_2297_	
0?-
&?#
!?
input_1?????????
? "%?"
?
0?????????
? ?
+__inference_custom_model_layer_call_fn_2311R	
0?-
&?#
!?
input_1?????????
? "???????????
"__inference_signature_wrapper_2215?	
*+./u?r
? 
k?h
1
training_input?
training_input
3
training_target ?
training_target""?

output_0?
output_0 ?
"__inference_signature_wrapper_2230d	
0?-
? 
&?#
!
inputs?
inputs"*?'
%
output_0?
output_0?
A__inference_test_in_layer_call_and_return_conditional_losses_2550\	
/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
&__inference_test_in_layer_call_fn_2540O	
/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_test_out_layer_call_and_return_conditional_losses_2569\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_test_out_layer_call_fn_2559O/?,
%?"
 ?
inputs?????????
? "???????????
__inference_training_2531m	
*+./U?R
K?H
F?C
?
training_input
 ?
training_target
? "? 