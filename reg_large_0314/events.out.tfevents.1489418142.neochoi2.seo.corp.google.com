       �K"	  ��1�Abrain.Event:2Q%a��     �/��	ȷ��1�A"��

global_step/Initializer/ConstConst*
dtype0	*
_class
loc:@global_step*
value	B	 R *
_output_shapes
: 
�
global_step
VariableV2*
	container *
_output_shapes
: *
dtype0	*
shape: *
_class
loc:@global_step*
shared_name 
�
global_step/AssignAssignglobal_stepglobal_step/Initializer/Const*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
T0	*
_output_shapes
: 
W
inputPlaceholder*
dtype0*
shape: *'
_output_shapes
:���������T
T
outputPlaceholder*
dtype0	*
shape: *#
_output_shapes
:���������
�
Kdnn/input_from_feature_columns/input_from_feature_columns/concat/concat_dimConst*
dtype0*
value	B :*
_output_shapes
: 
�
@dnn/input_from_feature_columns/input_from_feature_columns/concatIdentityinput*
T0*'
_output_shapes
:���������T
�
Adnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB"T   Q   *
_output_shapes
:
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *�DC�*
_output_shapes
: 
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *�DC>*
_output_shapes
: 
�
Idnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:TQ*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes
: 
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:TQ
�
;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:TQ
�
 dnn/hiddenlayer_0/weights/part_0
VariableV2*
	container *
_output_shapes

:TQ*
dtype0*
shape
:TQ*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
shared_name 
�
'dnn/hiddenlayer_0/weights/part_0/AssignAssign dnn/hiddenlayer_0/weights/part_0;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking(*
T0*
_output_shapes

:TQ
�
%dnn/hiddenlayer_0/weights/part_0/readIdentity dnn/hiddenlayer_0/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:TQ
�
1dnn/hiddenlayer_0/biases/part_0/Initializer/ConstConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
valueBQ*    *
_output_shapes
:Q
�
dnn/hiddenlayer_0/biases/part_0
VariableV2*
	container *
_output_shapes
:Q*
dtype0*
shape:Q*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
shared_name 
�
&dnn/hiddenlayer_0/biases/part_0/AssignAssigndnn/hiddenlayer_0/biases/part_01dnn/hiddenlayer_0/biases/part_0/Initializer/Const*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q
�
$dnn/hiddenlayer_0/biases/part_0/readIdentitydnn/hiddenlayer_0/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
T0*
_output_shapes
:Q
u
dnn/hiddenlayer_0/weightsIdentity%dnn/hiddenlayer_0/weights/part_0/read*
T0*
_output_shapes

:TQ
�
dnn/hiddenlayer_0/MatMulMatMul@dnn/input_from_feature_columns/input_from_feature_columns/concatdnn/hiddenlayer_0/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������Q
o
dnn/hiddenlayer_0/biasesIdentity$dnn/hiddenlayer_0/biases/part_0/read*
T0*
_output_shapes
:Q
�
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/biases*'
_output_shapes
:���������Q*
T0*
data_formatNHWC
y
$dnn/hiddenlayer_0/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*'
_output_shapes
:���������Q
W
zero_fraction/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
zero_fraction/EqualEqual$dnn/hiddenlayer_0/hiddenlayer_0/Reluzero_fraction/zero*
T0*'
_output_shapes
:���������Q
p
zero_fraction/CastCastzero_fraction/Equal*

DstT0*

SrcT0
*'
_output_shapes
:���������Q
d
zero_fraction/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
zero_fraction/MeanMeanzero_fraction/Castzero_fraction/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
.dnn/hiddenlayer_0_fraction_of_zero_values/tagsConst*
dtype0*:
value1B/ B)dnn/hiddenlayer_0_fraction_of_zero_values*
_output_shapes
: 
�
)dnn/hiddenlayer_0_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_0_fraction_of_zero_values/tagszero_fraction/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_0_activation/tagConst*
dtype0*-
value$B" Bdnn/hiddenlayer_0_activation*
_output_shapes
: 
�
dnn/hiddenlayer_0_activationHistogramSummary dnn/hiddenlayer_0_activation/tag$dnn/hiddenlayer_0/hiddenlayer_0/Relu*
T0*
_output_shapes
: 
�
Adnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB"Q   Q   *
_output_shapes
:
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *�E�*
_output_shapes
: 
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *�E>*
_output_shapes
: 
�
Idnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:QQ*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes
: 
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:QQ
�
;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:QQ
�
 dnn/hiddenlayer_1/weights/part_0
VariableV2*
	container *
_output_shapes

:QQ*
dtype0*
shape
:QQ*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
shared_name 
�
'dnn/hiddenlayer_1/weights/part_0/AssignAssign dnn/hiddenlayer_1/weights/part_0;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking(*
T0*
_output_shapes

:QQ
�
%dnn/hiddenlayer_1/weights/part_0/readIdentity dnn/hiddenlayer_1/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:QQ
�
1dnn/hiddenlayer_1/biases/part_0/Initializer/ConstConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
valueBQ*    *
_output_shapes
:Q
�
dnn/hiddenlayer_1/biases/part_0
VariableV2*
	container *
_output_shapes
:Q*
dtype0*
shape:Q*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
shared_name 
�
&dnn/hiddenlayer_1/biases/part_0/AssignAssigndnn/hiddenlayer_1/biases/part_01dnn/hiddenlayer_1/biases/part_0/Initializer/Const*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q
�
$dnn/hiddenlayer_1/biases/part_0/readIdentitydnn/hiddenlayer_1/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
T0*
_output_shapes
:Q
u
dnn/hiddenlayer_1/weightsIdentity%dnn/hiddenlayer_1/weights/part_0/read*
T0*
_output_shapes

:QQ
�
dnn/hiddenlayer_1/MatMulMatMul$dnn/hiddenlayer_0/hiddenlayer_0/Reludnn/hiddenlayer_1/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������Q
o
dnn/hiddenlayer_1/biasesIdentity$dnn/hiddenlayer_1/biases/part_0/read*
T0*
_output_shapes
:Q
�
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/biases*'
_output_shapes
:���������Q*
T0*
data_formatNHWC
y
$dnn/hiddenlayer_1/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:���������Q
Y
zero_fraction_1/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
zero_fraction_1/EqualEqual$dnn/hiddenlayer_1/hiddenlayer_1/Reluzero_fraction_1/zero*
T0*'
_output_shapes
:���������Q
t
zero_fraction_1/CastCastzero_fraction_1/Equal*

DstT0*

SrcT0
*'
_output_shapes
:���������Q
f
zero_fraction_1/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
zero_fraction_1/MeanMeanzero_fraction_1/Castzero_fraction_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
.dnn/hiddenlayer_1_fraction_of_zero_values/tagsConst*
dtype0*:
value1B/ B)dnn/hiddenlayer_1_fraction_of_zero_values*
_output_shapes
: 
�
)dnn/hiddenlayer_1_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_1_fraction_of_zero_values/tagszero_fraction_1/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_1_activation/tagConst*
dtype0*-
value$B" Bdnn/hiddenlayer_1_activation*
_output_shapes
: 
�
dnn/hiddenlayer_1_activationHistogramSummary dnn/hiddenlayer_1_activation/tag$dnn/hiddenlayer_1/hiddenlayer_1/Relu*
T0*
_output_shapes
: 
�
Adnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB"Q   1   *
_output_shapes
:
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB
 *��[�*
_output_shapes
: 
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB
 *��[>*
_output_shapes
: 
�
Idnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:Q1*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes
: 
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:Q1
�
;dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:Q1
�
 dnn/hiddenlayer_2/weights/part_0
VariableV2*
	container *
_output_shapes

:Q1*
dtype0*
shape
:Q1*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
shared_name 
�
'dnn/hiddenlayer_2/weights/part_0/AssignAssign dnn/hiddenlayer_2/weights/part_0;dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
use_locking(*
T0*
_output_shapes

:Q1
�
%dnn/hiddenlayer_2/weights/part_0/readIdentity dnn/hiddenlayer_2/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:Q1
�
1dnn/hiddenlayer_2/biases/part_0/Initializer/ConstConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
valueB1*    *
_output_shapes
:1
�
dnn/hiddenlayer_2/biases/part_0
VariableV2*
	container *
_output_shapes
:1*
dtype0*
shape:1*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
shared_name 
�
&dnn/hiddenlayer_2/biases/part_0/AssignAssigndnn/hiddenlayer_2/biases/part_01dnn/hiddenlayer_2/biases/part_0/Initializer/Const*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
use_locking(*
T0*
_output_shapes
:1
�
$dnn/hiddenlayer_2/biases/part_0/readIdentitydnn/hiddenlayer_2/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
T0*
_output_shapes
:1
u
dnn/hiddenlayer_2/weightsIdentity%dnn/hiddenlayer_2/weights/part_0/read*
T0*
_output_shapes

:Q1
�
dnn/hiddenlayer_2/MatMulMatMul$dnn/hiddenlayer_1/hiddenlayer_1/Reludnn/hiddenlayer_2/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������1
o
dnn/hiddenlayer_2/biasesIdentity$dnn/hiddenlayer_2/biases/part_0/read*
T0*
_output_shapes
:1
�
dnn/hiddenlayer_2/BiasAddBiasAdddnn/hiddenlayer_2/MatMuldnn/hiddenlayer_2/biases*'
_output_shapes
:���������1*
T0*
data_formatNHWC
y
$dnn/hiddenlayer_2/hiddenlayer_2/ReluReludnn/hiddenlayer_2/BiasAdd*
T0*'
_output_shapes
:���������1
Y
zero_fraction_2/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
zero_fraction_2/EqualEqual$dnn/hiddenlayer_2/hiddenlayer_2/Reluzero_fraction_2/zero*
T0*'
_output_shapes
:���������1
t
zero_fraction_2/CastCastzero_fraction_2/Equal*

DstT0*

SrcT0
*'
_output_shapes
:���������1
f
zero_fraction_2/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
zero_fraction_2/MeanMeanzero_fraction_2/Castzero_fraction_2/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
.dnn/hiddenlayer_2_fraction_of_zero_values/tagsConst*
dtype0*:
value1B/ B)dnn/hiddenlayer_2_fraction_of_zero_values*
_output_shapes
: 
�
)dnn/hiddenlayer_2_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_2_fraction_of_zero_values/tagszero_fraction_2/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_2_activation/tagConst*
dtype0*-
value$B" Bdnn/hiddenlayer_2_activation*
_output_shapes
: 
�
dnn/hiddenlayer_2_activationHistogramSummary dnn/hiddenlayer_2_activation/tag$dnn/hiddenlayer_2/hiddenlayer_2/Relu*
T0*
_output_shapes
: 
�
Adnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
valueB"1      *
_output_shapes
:
�
?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
valueB
 *iʑ�*
_output_shapes
: 
�
?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
valueB
 *iʑ>*
_output_shapes
: 
�
Idnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:1*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0
�
?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
T0*
_output_shapes
: 
�
?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
T0*
_output_shapes

:1
�
;dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
T0*
_output_shapes

:1
�
 dnn/hiddenlayer_3/weights/part_0
VariableV2*
	container *
_output_shapes

:1*
dtype0*
shape
:1*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
shared_name 
�
'dnn/hiddenlayer_3/weights/part_0/AssignAssign dnn/hiddenlayer_3/weights/part_0;dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
use_locking(*
T0*
_output_shapes

:1
�
%dnn/hiddenlayer_3/weights/part_0/readIdentity dnn/hiddenlayer_3/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
T0*
_output_shapes

:1
�
1dnn/hiddenlayer_3/biases/part_0/Initializer/ConstConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
valueB*    *
_output_shapes
:
�
dnn/hiddenlayer_3/biases/part_0
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
shared_name 
�
&dnn/hiddenlayer_3/biases/part_0/AssignAssigndnn/hiddenlayer_3/biases/part_01dnn/hiddenlayer_3/biases/part_0/Initializer/Const*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
use_locking(*
T0*
_output_shapes
:
�
$dnn/hiddenlayer_3/biases/part_0/readIdentitydnn/hiddenlayer_3/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
T0*
_output_shapes
:
u
dnn/hiddenlayer_3/weightsIdentity%dnn/hiddenlayer_3/weights/part_0/read*
T0*
_output_shapes

:1
�
dnn/hiddenlayer_3/MatMulMatMul$dnn/hiddenlayer_2/hiddenlayer_2/Reludnn/hiddenlayer_3/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
o
dnn/hiddenlayer_3/biasesIdentity$dnn/hiddenlayer_3/biases/part_0/read*
T0*
_output_shapes
:
�
dnn/hiddenlayer_3/BiasAddBiasAdddnn/hiddenlayer_3/MatMuldnn/hiddenlayer_3/biases*'
_output_shapes
:���������*
T0*
data_formatNHWC
y
$dnn/hiddenlayer_3/hiddenlayer_3/ReluReludnn/hiddenlayer_3/BiasAdd*
T0*'
_output_shapes
:���������
Y
zero_fraction_3/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
zero_fraction_3/EqualEqual$dnn/hiddenlayer_3/hiddenlayer_3/Reluzero_fraction_3/zero*
T0*'
_output_shapes
:���������
t
zero_fraction_3/CastCastzero_fraction_3/Equal*

DstT0*

SrcT0
*'
_output_shapes
:���������
f
zero_fraction_3/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
zero_fraction_3/MeanMeanzero_fraction_3/Castzero_fraction_3/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
.dnn/hiddenlayer_3_fraction_of_zero_values/tagsConst*
dtype0*:
value1B/ B)dnn/hiddenlayer_3_fraction_of_zero_values*
_output_shapes
: 
�
)dnn/hiddenlayer_3_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_3_fraction_of_zero_values/tagszero_fraction_3/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_3_activation/tagConst*
dtype0*-
value$B" Bdnn/hiddenlayer_3_activation*
_output_shapes
: 
�
dnn/hiddenlayer_3_activationHistogramSummary dnn/hiddenlayer_3_activation/tag$dnn/hiddenlayer_3/hiddenlayer_3/Relu*
T0*
_output_shapes
: 
�
:dnn/logits/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB"      *
_output_shapes
:
�
8dnn/logits/weights/part_0/Initializer/random_uniform/minConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *����*
_output_shapes
: 
�
8dnn/logits/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *���>*
_output_shapes
: 
�
Bdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniform:dnn/logits/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:*
dtype0*
seed2 *

seed *
T0*,
_class"
 loc:@dnn/logits/weights/part_0
�
8dnn/logits/weights/part_0/Initializer/random_uniform/subSub8dnn/logits/weights/part_0/Initializer/random_uniform/max8dnn/logits/weights/part_0/Initializer/random_uniform/min*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes
: 
�
8dnn/logits/weights/part_0/Initializer/random_uniform/mulMulBdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniform8dnn/logits/weights/part_0/Initializer/random_uniform/sub*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

:
�
4dnn/logits/weights/part_0/Initializer/random_uniformAdd8dnn/logits/weights/part_0/Initializer/random_uniform/mul8dnn/logits/weights/part_0/Initializer/random_uniform/min*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

:
�
dnn/logits/weights/part_0
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*,
_class"
 loc:@dnn/logits/weights/part_0*
shared_name 
�
 dnn/logits/weights/part_0/AssignAssigndnn/logits/weights/part_04dnn/logits/weights/part_0/Initializer/random_uniform*
validate_shape(*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking(*
T0*
_output_shapes

:
�
dnn/logits/weights/part_0/readIdentitydnn/logits/weights/part_0*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

:
�
*dnn/logits/biases/part_0/Initializer/ConstConst*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
valueB*    *
_output_shapes
:
�
dnn/logits/biases/part_0
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*+
_class!
loc:@dnn/logits/biases/part_0*
shared_name 
�
dnn/logits/biases/part_0/AssignAssigndnn/logits/biases/part_0*dnn/logits/biases/part_0/Initializer/Const*
validate_shape(*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking(*
T0*
_output_shapes
:
�
dnn/logits/biases/part_0/readIdentitydnn/logits/biases/part_0*+
_class!
loc:@dnn/logits/biases/part_0*
T0*
_output_shapes
:
g
dnn/logits/weightsIdentitydnn/logits/weights/part_0/read*
T0*
_output_shapes

:
�
dnn/logits/MatMulMatMul$dnn/hiddenlayer_3/hiddenlayer_3/Reludnn/logits/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
a
dnn/logits/biasesIdentitydnn/logits/biases/part_0/read*
T0*
_output_shapes
:
�
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/biases*'
_output_shapes
:���������*
T0*
data_formatNHWC
Y
zero_fraction_4/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
z
zero_fraction_4/EqualEqualdnn/logits/BiasAddzero_fraction_4/zero*
T0*'
_output_shapes
:���������
t
zero_fraction_4/CastCastzero_fraction_4/Equal*

DstT0*

SrcT0
*'
_output_shapes
:���������
f
zero_fraction_4/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
zero_fraction_4/MeanMeanzero_fraction_4/Castzero_fraction_4/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
'dnn/logits_fraction_of_zero_values/tagsConst*
dtype0*3
value*B( B"dnn/logits_fraction_of_zero_values*
_output_shapes
: 
�
"dnn/logits_fraction_of_zero_valuesScalarSummary'dnn/logits_fraction_of_zero_values/tagszero_fraction_4/Mean*
T0*
_output_shapes
: 
o
dnn/logits_activation/tagConst*
dtype0*&
valueB Bdnn/logits_activation*
_output_shapes
: 
y
dnn/logits_activationHistogramSummarydnn/logits_activation/tagdnn/logits/BiasAdd*
T0*
_output_shapes
: 
v
predictions/scoresSqueezednn/logits/BiasAdd*
squeeze_dims
*
T0*#
_output_shapes
:���������
x
.training_loss/mean_squared_loss/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
�
*training_loss/mean_squared_loss/ExpandDims
ExpandDimsoutput.training_loss/mean_squared_loss/ExpandDims/dim*

Tdim0*
T0	*'
_output_shapes
:���������
�
'training_loss/mean_squared_loss/ToFloatCast*training_loss/mean_squared_loss/ExpandDims*

DstT0*

SrcT0	*'
_output_shapes
:���������
�
#training_loss/mean_squared_loss/subSubdnn/logits/BiasAdd'training_loss/mean_squared_loss/ToFloat*
T0*'
_output_shapes
:���������
�
training_loss/mean_squared_lossSquare#training_loss/mean_squared_loss/sub*
T0*'
_output_shapes
:���������
d
training_loss/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
training_lossMeantraining_loss/mean_squared_losstraining_loss/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
e
 training_loss/ScalarSummary/tagsConst*
dtype0*
valueB
 Bloss*
_output_shapes
: 
~
training_loss/ScalarSummaryScalarSummary training_loss/ScalarSummary/tagstraining_loss*
T0*
_output_shapes
: 
�
#dnn/learning_rate/Initializer/ConstConst*
dtype0*$
_class
loc:@dnn/learning_rate*
valueB
 *��L=*
_output_shapes
: 
�
dnn/learning_rate
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *$
_class
loc:@dnn/learning_rate*
shared_name 
�
dnn/learning_rate/AssignAssigndnn/learning_rate#dnn/learning_rate/Initializer/Const*
validate_shape(*$
_class
loc:@dnn/learning_rate*
use_locking(*
T0*
_output_shapes
: 
|
dnn/learning_rate/readIdentitydnn/learning_rate*$
_class
loc:@dnn/learning_rate*
T0*
_output_shapes
: 
_
train_op/dnn/gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
a
train_op/dnn/gradients/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
train_op/dnn/gradients/FillFilltrain_op/dnn/gradients/Shapetrain_op/dnn/gradients/Const*
T0*
_output_shapes
: 
�
7train_op/dnn/gradients/training_loss_grad/Reshape/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
�
1train_op/dnn/gradients/training_loss_grad/ReshapeReshapetrain_op/dnn/gradients/Fill7train_op/dnn/gradients/training_loss_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
�
/train_op/dnn/gradients/training_loss_grad/ShapeShapetraining_loss/mean_squared_loss*
out_type0*
T0*
_output_shapes
:
�
.train_op/dnn/gradients/training_loss_grad/TileTile1train_op/dnn/gradients/training_loss_grad/Reshape/train_op/dnn/gradients/training_loss_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
�
1train_op/dnn/gradients/training_loss_grad/Shape_1Shapetraining_loss/mean_squared_loss*
out_type0*
T0*
_output_shapes
:
t
1train_op/dnn/gradients/training_loss_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
y
/train_op/dnn/gradients/training_loss_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
.train_op/dnn/gradients/training_loss_grad/ProdProd1train_op/dnn/gradients/training_loss_grad/Shape_1/train_op/dnn/gradients/training_loss_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
{
1train_op/dnn/gradients/training_loss_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
�
0train_op/dnn/gradients/training_loss_grad/Prod_1Prod1train_op/dnn/gradients/training_loss_grad/Shape_21train_op/dnn/gradients/training_loss_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
u
3train_op/dnn/gradients/training_loss_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
1train_op/dnn/gradients/training_loss_grad/MaximumMaximum0train_op/dnn/gradients/training_loss_grad/Prod_13train_op/dnn/gradients/training_loss_grad/Maximum/y*
T0*
_output_shapes
: 
�
2train_op/dnn/gradients/training_loss_grad/floordivFloorDiv.train_op/dnn/gradients/training_loss_grad/Prod1train_op/dnn/gradients/training_loss_grad/Maximum*
T0*
_output_shapes
: 
�
.train_op/dnn/gradients/training_loss_grad/CastCast2train_op/dnn/gradients/training_loss_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 
�
1train_op/dnn/gradients/training_loss_grad/truedivRealDiv.train_op/dnn/gradients/training_loss_grad/Tile.train_op/dnn/gradients/training_loss_grad/Cast*
T0*'
_output_shapes
:���������
�
Atrain_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul/xConst2^train_op/dnn/gradients/training_loss_grad/truediv*
dtype0*
valueB
 *   @*
_output_shapes
: 
�
?train_op/dnn/gradients/training_loss/mean_squared_loss_grad/mulMulAtrain_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul/x#training_loss/mean_squared_loss/sub*
T0*'
_output_shapes
:���������
�
Atrain_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul_1Mul1train_op/dnn/gradients/training_loss_grad/truediv?train_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul*
T0*'
_output_shapes
:���������
�
Etrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/ShapeShapednn/logits/BiasAdd*
out_type0*
T0*
_output_shapes
:
�
Gtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Shape_1Shape'training_loss/mean_squared_loss/ToFloat*
out_type0*
T0*
_output_shapes
:
�
Utrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgsEtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/ShapeGtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ctrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/SumSumAtrain_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul_1Utrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Gtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/ReshapeReshapeCtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/SumEtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
Etrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Sum_1SumAtrain_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul_1Wtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Ctrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/NegNegEtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Sum_1*
T0*
_output_shapes
:
�
Itrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape_1ReshapeCtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/NegGtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
Ptrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/group_depsNoOpH^train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/ReshapeJ^train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape_1
�
Xtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/control_dependencyIdentityGtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/ReshapeQ^train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/group_deps*Z
_classP
NLloc:@train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape*
T0*'
_output_shapes
:���������
�
Ztrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/control_dependency_1IdentityItrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape_1Q^train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/group_deps*\
_classR
PNloc:@train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
:train_op/dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGradBiasAddGradXtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/control_dependency*
_output_shapes
:*
T0*
data_formatNHWC
�
?train_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/group_depsNoOpY^train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/control_dependency;^train_op/dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGrad
�
Gtrain_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependencyIdentityXtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/control_dependency@^train_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/group_deps*Z
_classP
NLloc:@train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape*
T0*'
_output_shapes
:���������
�
Itrain_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependency_1Identity:train_op/dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGrad@^train_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/group_deps*M
_classC
A?loc:@train_op/dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
�
4train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMulMatMulGtrain_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependencydnn/logits/weights*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������
�
6train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul_1MatMul$dnn/hiddenlayer_3/hiddenlayer_3/ReluGtrain_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:
�
>train_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/group_depsNoOp5^train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul7^train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul_1
�
Ftrain_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependencyIdentity4train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul?^train_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/group_deps*G
_class=
;9loc:@train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������
�
Htrain_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependency_1Identity6train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul_1?^train_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/group_deps*I
_class?
=;loc:@train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
�
Itrain_op/dnn/gradients/dnn/hiddenlayer_3/hiddenlayer_3/Relu_grad/ReluGradReluGradFtrain_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependency$dnn/hiddenlayer_3/hiddenlayer_3/Relu*
T0*'
_output_shapes
:���������
�
Atrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/BiasAddGradBiasAddGradItrain_op/dnn/gradients/dnn/hiddenlayer_3/hiddenlayer_3/Relu_grad/ReluGrad*
_output_shapes
:*
T0*
data_formatNHWC
�
Ftrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/group_depsNoOpJ^train_op/dnn/gradients/dnn/hiddenlayer_3/hiddenlayer_3/Relu_grad/ReluGradB^train_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/BiasAddGrad
�
Ntrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/control_dependencyIdentityItrain_op/dnn/gradients/dnn/hiddenlayer_3/hiddenlayer_3/Relu_grad/ReluGradG^train_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@train_op/dnn/gradients/dnn/hiddenlayer_3/hiddenlayer_3/Relu_grad/ReluGrad*
T0*'
_output_shapes
:���������
�
Ptrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/control_dependency_1IdentityAtrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/BiasAddGradG^train_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/group_deps*T
_classJ
HFloc:@train_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
�
;train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMulMatMulNtrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/control_dependencydnn/hiddenlayer_3/weights*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������1
�
=train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul_1MatMul$dnn/hiddenlayer_2/hiddenlayer_2/ReluNtrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:1
�
Etrain_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/group_depsNoOp<^train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul>^train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul_1
�
Mtrain_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/control_dependencyIdentity;train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMulF^train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������1
�
Otrain_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/control_dependency_1Identity=train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul_1F^train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

:1
�
Itrain_op/dnn/gradients/dnn/hiddenlayer_2/hiddenlayer_2/Relu_grad/ReluGradReluGradMtrain_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/control_dependency$dnn/hiddenlayer_2/hiddenlayer_2/Relu*
T0*'
_output_shapes
:���������1
�
Atrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/BiasAddGradBiasAddGradItrain_op/dnn/gradients/dnn/hiddenlayer_2/hiddenlayer_2/Relu_grad/ReluGrad*
_output_shapes
:1*
T0*
data_formatNHWC
�
Ftrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/group_depsNoOpJ^train_op/dnn/gradients/dnn/hiddenlayer_2/hiddenlayer_2/Relu_grad/ReluGradB^train_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/BiasAddGrad
�
Ntrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/control_dependencyIdentityItrain_op/dnn/gradients/dnn/hiddenlayer_2/hiddenlayer_2/Relu_grad/ReluGradG^train_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@train_op/dnn/gradients/dnn/hiddenlayer_2/hiddenlayer_2/Relu_grad/ReluGrad*
T0*'
_output_shapes
:���������1
�
Ptrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/control_dependency_1IdentityAtrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/BiasAddGradG^train_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/group_deps*T
_classJ
HFloc:@train_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:1
�
;train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMulMatMulNtrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/control_dependencydnn/hiddenlayer_2/weights*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������Q
�
=train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul_1MatMul$dnn/hiddenlayer_1/hiddenlayer_1/ReluNtrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:Q1
�
Etrain_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/group_depsNoOp<^train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul>^train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul_1
�
Mtrain_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/control_dependencyIdentity;train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMulF^train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������Q
�
Otrain_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/control_dependency_1Identity=train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul_1F^train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:Q1
�
Itrain_op/dnn/gradients/dnn/hiddenlayer_1/hiddenlayer_1/Relu_grad/ReluGradReluGradMtrain_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/control_dependency$dnn/hiddenlayer_1/hiddenlayer_1/Relu*
T0*'
_output_shapes
:���������Q
�
Atrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGradBiasAddGradItrain_op/dnn/gradients/dnn/hiddenlayer_1/hiddenlayer_1/Relu_grad/ReluGrad*
_output_shapes
:Q*
T0*
data_formatNHWC
�
Ftrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/group_depsNoOpJ^train_op/dnn/gradients/dnn/hiddenlayer_1/hiddenlayer_1/Relu_grad/ReluGradB^train_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGrad
�
Ntrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependencyIdentityItrain_op/dnn/gradients/dnn/hiddenlayer_1/hiddenlayer_1/Relu_grad/ReluGradG^train_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@train_op/dnn/gradients/dnn/hiddenlayer_1/hiddenlayer_1/Relu_grad/ReluGrad*
T0*'
_output_shapes
:���������Q
�
Ptrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependency_1IdentityAtrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGradG^train_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/group_deps*T
_classJ
HFloc:@train_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:Q
�
;train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMulMatMulNtrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependencydnn/hiddenlayer_1/weights*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������Q
�
=train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1MatMul$dnn/hiddenlayer_0/hiddenlayer_0/ReluNtrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:QQ
�
Etrain_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/group_depsNoOp<^train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul>^train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1
�
Mtrain_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependencyIdentity;train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMulF^train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������Q
�
Otrain_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependency_1Identity=train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1F^train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:QQ
�
Itrain_op/dnn/gradients/dnn/hiddenlayer_0/hiddenlayer_0/Relu_grad/ReluGradReluGradMtrain_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependency$dnn/hiddenlayer_0/hiddenlayer_0/Relu*
T0*'
_output_shapes
:���������Q
�
Atrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGradBiasAddGradItrain_op/dnn/gradients/dnn/hiddenlayer_0/hiddenlayer_0/Relu_grad/ReluGrad*
_output_shapes
:Q*
T0*
data_formatNHWC
�
Ftrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/group_depsNoOpJ^train_op/dnn/gradients/dnn/hiddenlayer_0/hiddenlayer_0/Relu_grad/ReluGradB^train_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGrad
�
Ntrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependencyIdentityItrain_op/dnn/gradients/dnn/hiddenlayer_0/hiddenlayer_0/Relu_grad/ReluGradG^train_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@train_op/dnn/gradients/dnn/hiddenlayer_0/hiddenlayer_0/Relu_grad/ReluGrad*
T0*'
_output_shapes
:���������Q
�
Ptrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependency_1IdentityAtrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGradG^train_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/group_deps*T
_classJ
HFloc:@train_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:Q
�
;train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMulMatMulNtrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependencydnn/hiddenlayer_0/weights*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������T
�
=train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1MatMul@dnn/input_from_feature_columns/input_from_feature_columns/concatNtrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:TQ
�
Etrain_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/group_depsNoOp<^train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul>^train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1
�
Mtrain_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/control_dependencyIdentity;train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMulF^train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������T
�
Otrain_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/control_dependency_1Identity=train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1F^train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1*
T0*
_output_shapes

:TQ
�
train_op/dnn/ConstConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueBTQ*���=*
_output_shapes

:TQ
�
,dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad
VariableV2*
	container *
_output_shapes

:TQ*
dtype0*
shape
:TQ*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
shared_name 
�
3dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad/AssignAssign,dnn/dnn/hiddenlayer_0/weights/part_0/Adagradtrain_op/dnn/Const*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking(*
T0*
_output_shapes

:TQ
�
1dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad/readIdentity,dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:TQ
�
train_op/dnn/Const_1Const*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
valueBQ*���=*
_output_shapes
:Q
�
+dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad
VariableV2*
	container *
_output_shapes
:Q*
dtype0*
shape:Q*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
shared_name 
�
2dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad/AssignAssign+dnn/dnn/hiddenlayer_0/biases/part_0/Adagradtrain_op/dnn/Const_1*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q
�
0dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad/readIdentity+dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
T0*
_output_shapes
:Q
�
train_op/dnn/Const_2Const*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueBQQ*���=*
_output_shapes

:QQ
�
,dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad
VariableV2*
	container *
_output_shapes

:QQ*
dtype0*
shape
:QQ*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
shared_name 
�
3dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad/AssignAssign,dnn/dnn/hiddenlayer_1/weights/part_0/Adagradtrain_op/dnn/Const_2*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking(*
T0*
_output_shapes

:QQ
�
1dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad/readIdentity,dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:QQ
�
train_op/dnn/Const_3Const*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
valueBQ*���=*
_output_shapes
:Q
�
+dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad
VariableV2*
	container *
_output_shapes
:Q*
dtype0*
shape:Q*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
shared_name 
�
2dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad/AssignAssign+dnn/dnn/hiddenlayer_1/biases/part_0/Adagradtrain_op/dnn/Const_3*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q
�
0dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad/readIdentity+dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
T0*
_output_shapes
:Q
�
train_op/dnn/Const_4Const*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueBQ1*���=*
_output_shapes

:Q1
�
,dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad
VariableV2*
	container *
_output_shapes

:Q1*
dtype0*
shape
:Q1*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
shared_name 
�
3dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad/AssignAssign,dnn/dnn/hiddenlayer_2/weights/part_0/Adagradtrain_op/dnn/Const_4*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
use_locking(*
T0*
_output_shapes

:Q1
�
1dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad/readIdentity,dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:Q1
�
train_op/dnn/Const_5Const*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
valueB1*���=*
_output_shapes
:1
�
+dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad
VariableV2*
	container *
_output_shapes
:1*
dtype0*
shape:1*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
shared_name 
�
2dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad/AssignAssign+dnn/dnn/hiddenlayer_2/biases/part_0/Adagradtrain_op/dnn/Const_5*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
use_locking(*
T0*
_output_shapes
:1
�
0dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad/readIdentity+dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
T0*
_output_shapes
:1
�
train_op/dnn/Const_6Const*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
valueB1*���=*
_output_shapes

:1
�
,dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad
VariableV2*
	container *
_output_shapes

:1*
dtype0*
shape
:1*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
shared_name 
�
3dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad/AssignAssign,dnn/dnn/hiddenlayer_3/weights/part_0/Adagradtrain_op/dnn/Const_6*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
use_locking(*
T0*
_output_shapes

:1
�
1dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad/readIdentity,dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
T0*
_output_shapes

:1
�
train_op/dnn/Const_7Const*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
valueB*���=*
_output_shapes
:
�
+dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
shared_name 
�
2dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad/AssignAssign+dnn/dnn/hiddenlayer_3/biases/part_0/Adagradtrain_op/dnn/Const_7*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
use_locking(*
T0*
_output_shapes
:
�
0dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad/readIdentity+dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
T0*
_output_shapes
:
�
train_op/dnn/Const_8Const*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB*���=*
_output_shapes

:
�
%dnn/dnn/logits/weights/part_0/Adagrad
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*,
_class"
 loc:@dnn/logits/weights/part_0*
shared_name 
�
,dnn/dnn/logits/weights/part_0/Adagrad/AssignAssign%dnn/dnn/logits/weights/part_0/Adagradtrain_op/dnn/Const_8*
validate_shape(*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking(*
T0*
_output_shapes

:
�
*dnn/dnn/logits/weights/part_0/Adagrad/readIdentity%dnn/dnn/logits/weights/part_0/Adagrad*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

:
�
train_op/dnn/Const_9Const*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
valueB*���=*
_output_shapes
:
�
$dnn/dnn/logits/biases/part_0/Adagrad
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*+
_class!
loc:@dnn/logits/biases/part_0*
shared_name 
�
+dnn/dnn/logits/biases/part_0/Adagrad/AssignAssign$dnn/dnn/logits/biases/part_0/Adagradtrain_op/dnn/Const_9*
validate_shape(*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking(*
T0*
_output_shapes
:
�
)dnn/dnn/logits/biases/part_0/Adagrad/readIdentity$dnn/dnn/logits/biases/part_0/Adagrad*+
_class!
loc:@dnn/logits/biases/part_0*
T0*
_output_shapes
:
�
Gtrain_op/dnn/train/update_dnn/hiddenlayer_0/weights/part_0/ApplyAdagradApplyAdagrad dnn/hiddenlayer_0/weights/part_0,dnn/dnn/hiddenlayer_0/weights/part_0/Adagraddnn/learning_rate/readOtrain_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/control_dependency_1*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking( *
T0*
_output_shapes

:TQ
�
Ftrain_op/dnn/train/update_dnn/hiddenlayer_0/biases/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_0/biases/part_0+dnn/dnn/hiddenlayer_0/biases/part_0/Adagraddnn/learning_rate/readPtrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependency_1*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking( *
T0*
_output_shapes
:Q
�
Gtrain_op/dnn/train/update_dnn/hiddenlayer_1/weights/part_0/ApplyAdagradApplyAdagrad dnn/hiddenlayer_1/weights/part_0,dnn/dnn/hiddenlayer_1/weights/part_0/Adagraddnn/learning_rate/readOtrain_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependency_1*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking( *
T0*
_output_shapes

:QQ
�
Ftrain_op/dnn/train/update_dnn/hiddenlayer_1/biases/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_1/biases/part_0+dnn/dnn/hiddenlayer_1/biases/part_0/Adagraddnn/learning_rate/readPtrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependency_1*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking( *
T0*
_output_shapes
:Q
�
Gtrain_op/dnn/train/update_dnn/hiddenlayer_2/weights/part_0/ApplyAdagradApplyAdagrad dnn/hiddenlayer_2/weights/part_0,dnn/dnn/hiddenlayer_2/weights/part_0/Adagraddnn/learning_rate/readOtrain_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/control_dependency_1*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
use_locking( *
T0*
_output_shapes

:Q1
�
Ftrain_op/dnn/train/update_dnn/hiddenlayer_2/biases/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_2/biases/part_0+dnn/dnn/hiddenlayer_2/biases/part_0/Adagraddnn/learning_rate/readPtrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/control_dependency_1*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
use_locking( *
T0*
_output_shapes
:1
�
Gtrain_op/dnn/train/update_dnn/hiddenlayer_3/weights/part_0/ApplyAdagradApplyAdagrad dnn/hiddenlayer_3/weights/part_0,dnn/dnn/hiddenlayer_3/weights/part_0/Adagraddnn/learning_rate/readOtrain_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/control_dependency_1*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
use_locking( *
T0*
_output_shapes

:1
�
Ftrain_op/dnn/train/update_dnn/hiddenlayer_3/biases/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_3/biases/part_0+dnn/dnn/hiddenlayer_3/biases/part_0/Adagraddnn/learning_rate/readPtrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/control_dependency_1*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
use_locking( *
T0*
_output_shapes
:
�
@train_op/dnn/train/update_dnn/logits/weights/part_0/ApplyAdagradApplyAdagraddnn/logits/weights/part_0%dnn/dnn/logits/weights/part_0/Adagraddnn/learning_rate/readHtrain_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependency_1*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking( *
T0*
_output_shapes

:
�
?train_op/dnn/train/update_dnn/logits/biases/part_0/ApplyAdagradApplyAdagraddnn/logits/biases/part_0$dnn/dnn/logits/biases/part_0/Adagraddnn/learning_rate/readItrain_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependency_1*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking( *
T0*
_output_shapes
:
�
train_op/dnn/train/updateNoOpH^train_op/dnn/train/update_dnn/hiddenlayer_0/weights/part_0/ApplyAdagradG^train_op/dnn/train/update_dnn/hiddenlayer_0/biases/part_0/ApplyAdagradH^train_op/dnn/train/update_dnn/hiddenlayer_1/weights/part_0/ApplyAdagradG^train_op/dnn/train/update_dnn/hiddenlayer_1/biases/part_0/ApplyAdagradH^train_op/dnn/train/update_dnn/hiddenlayer_2/weights/part_0/ApplyAdagradG^train_op/dnn/train/update_dnn/hiddenlayer_2/biases/part_0/ApplyAdagradH^train_op/dnn/train/update_dnn/hiddenlayer_3/weights/part_0/ApplyAdagradG^train_op/dnn/train/update_dnn/hiddenlayer_3/biases/part_0/ApplyAdagradA^train_op/dnn/train/update_dnn/logits/weights/part_0/ApplyAdagrad@^train_op/dnn/train/update_dnn/logits/biases/part_0/ApplyAdagrad
�
train_op/dnn/train/valueConst^train_op/dnn/train/update*
dtype0	*
_class
loc:@global_step*
value	B	 R*
_output_shapes
: 
�
train_op/dnn/train	AssignAddglobal_steptrain_op/dnn/train/value*
_class
loc:@global_step*
use_locking( *
T0	*
_output_shapes
: 
�
train_op/dnn/control_dependencyIdentitytraining_loss^train_op/dnn/train* 
_class
loc:@training_loss*
T0*
_output_shapes
: 
r
(metrics/mean_squared_loss/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
�
$metrics/mean_squared_loss/ExpandDims
ExpandDimsoutput(metrics/mean_squared_loss/ExpandDims/dim*

Tdim0*
T0	*'
_output_shapes
:���������
t
*metrics/mean_squared_loss/ExpandDims_1/dimConst*
dtype0*
valueB:*
_output_shapes
:
�
&metrics/mean_squared_loss/ExpandDims_1
ExpandDimspredictions/scores*metrics/mean_squared_loss/ExpandDims_1/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
!metrics/mean_squared_loss/ToFloatCast$metrics/mean_squared_loss/ExpandDims*

DstT0*

SrcT0	*'
_output_shapes
:���������
�
metrics/mean_squared_loss/subSub&metrics/mean_squared_loss/ExpandDims_1!metrics/mean_squared_loss/ToFloat*
T0*'
_output_shapes
:���������
t
metrics/mean_squared_lossSquaremetrics/mean_squared_loss/sub*
T0*'
_output_shapes
:���������
h
metrics/eval_loss/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
metrics/eval_lossMeanmetrics/mean_squared_lossmetrics/eval_loss/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
W
metrics/mean/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/total
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
�
metrics/mean/total/AssignAssignmetrics/mean/totalmetrics/mean/zeros*
validate_shape(*%
_class
loc:@metrics/mean/total*
use_locking(*
T0*
_output_shapes
: 

metrics/mean/total/readIdentitymetrics/mean/total*%
_class
loc:@metrics/mean/total*
T0*
_output_shapes
: 
Y
metrics/mean/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/count
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
�
metrics/mean/count/AssignAssignmetrics/mean/countmetrics/mean/zeros_1*
validate_shape(*%
_class
loc:@metrics/mean/count*
use_locking(*
T0*
_output_shapes
: 

metrics/mean/count/readIdentitymetrics/mean/count*%
_class
loc:@metrics/mean/count*
T0*
_output_shapes
: 
S
metrics/mean/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
a
metrics/mean/ToFloat_1Castmetrics/mean/Size*

DstT0*

SrcT0*
_output_shapes
: 
U
metrics/mean/ConstConst*
dtype0*
valueB *
_output_shapes
: 
|
metrics/mean/SumSummetrics/eval_lossmetrics/mean/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
metrics/mean/AssignAdd	AssignAddmetrics/mean/totalmetrics/mean/Sum*%
_class
loc:@metrics/mean/total*
use_locking( *
T0*
_output_shapes
: 
�
metrics/mean/AssignAdd_1	AssignAddmetrics/mean/countmetrics/mean/ToFloat_1*%
_class
loc:@metrics/mean/count*
use_locking( *
T0*
_output_shapes
: 
[
metrics/mean/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
q
metrics/mean/GreaterGreatermetrics/mean/count/readmetrics/mean/Greater/y*
T0*
_output_shapes
: 
r
metrics/mean/truedivRealDivmetrics/mean/total/readmetrics/mean/count/read*
T0*
_output_shapes
: 
Y
metrics/mean/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 

metrics/mean/valueSelectmetrics/mean/Greatermetrics/mean/truedivmetrics/mean/value/e*
T0*
_output_shapes
: 
]
metrics/mean/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/Greater_1Greatermetrics/mean/AssignAdd_1metrics/mean/Greater_1/y*
T0*
_output_shapes
: 
t
metrics/mean/truediv_1RealDivmetrics/mean/AssignAddmetrics/mean/AssignAdd_1*
T0*
_output_shapes
: 
]
metrics/mean/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
metrics/mean/update_opSelectmetrics/mean/Greater_1metrics/mean/truediv_1metrics/mean/update_op/e*
T0*
_output_shapes
: "Z	
G�(     ���	[��1�AJ��
��
9
Add
x"T
y"T
z"T"
Ttype:
2	
�
ApplyAdagrad
var"T�
accum"T�
lr"T	
grad"T
out"T�"
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
p
	AssignAdd
ref"T�

value"T

output_ref"T�"
Ttype:
2	"
use_lockingbool( 
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
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
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
A
Equal
x"T
y"T
z
"
Ttype:
2	
�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
:
Greater
x"T
y"T
z
"
Ttype:
2		
S
HistogramSummary
tag
values"T
summary"
Ttype0:
2		
.
Identity

input"T
output"T"	
Ttype
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
<
Mul
x"T
y"T
z"T"
Ttype:
2	�
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
A
Placeholder
output"dtype"
dtypetype"
shapeshape: 
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
0
Square
x"T
y"T"
Ttype:
	2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
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
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.0.12v1.0.0-65-g4763edf-dirty��

global_step/Initializer/ConstConst*
dtype0	*
_class
loc:@global_step*
value	B	 R *
_output_shapes
: 
�
global_step
VariableV2*
	container *
_output_shapes
: *
dtype0	*
shape: *
_class
loc:@global_step*
shared_name 
�
global_step/AssignAssignglobal_stepglobal_step/Initializer/Const*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
T0	*
_output_shapes
: 
W
inputPlaceholder*
dtype0*
shape: *'
_output_shapes
:���������T
T
outputPlaceholder*
dtype0	*
shape: *#
_output_shapes
:���������
�
Kdnn/input_from_feature_columns/input_from_feature_columns/concat/concat_dimConst*
dtype0*
value	B :*
_output_shapes
: 
�
@dnn/input_from_feature_columns/input_from_feature_columns/concatIdentityinput*
T0*'
_output_shapes
:���������T
�
Adnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB"T   Q   *
_output_shapes
:
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *�DC�*
_output_shapes
: 
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *�DC>*
_output_shapes
: 
�
Idnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:TQ*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes
: 
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:TQ
�
;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:TQ
�
 dnn/hiddenlayer_0/weights/part_0
VariableV2*
	container *
_output_shapes

:TQ*
dtype0*
shape
:TQ*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
shared_name 
�
'dnn/hiddenlayer_0/weights/part_0/AssignAssign dnn/hiddenlayer_0/weights/part_0;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking(*
T0*
_output_shapes

:TQ
�
%dnn/hiddenlayer_0/weights/part_0/readIdentity dnn/hiddenlayer_0/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:TQ
�
1dnn/hiddenlayer_0/biases/part_0/Initializer/ConstConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
valueBQ*    *
_output_shapes
:Q
�
dnn/hiddenlayer_0/biases/part_0
VariableV2*
	container *
_output_shapes
:Q*
dtype0*
shape:Q*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
shared_name 
�
&dnn/hiddenlayer_0/biases/part_0/AssignAssigndnn/hiddenlayer_0/biases/part_01dnn/hiddenlayer_0/biases/part_0/Initializer/Const*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q
�
$dnn/hiddenlayer_0/biases/part_0/readIdentitydnn/hiddenlayer_0/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
T0*
_output_shapes
:Q
u
dnn/hiddenlayer_0/weightsIdentity%dnn/hiddenlayer_0/weights/part_0/read*
T0*
_output_shapes

:TQ
�
dnn/hiddenlayer_0/MatMulMatMul@dnn/input_from_feature_columns/input_from_feature_columns/concatdnn/hiddenlayer_0/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������Q
o
dnn/hiddenlayer_0/biasesIdentity$dnn/hiddenlayer_0/biases/part_0/read*
T0*
_output_shapes
:Q
�
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/biases*
data_formatNHWC*
T0*'
_output_shapes
:���������Q
y
$dnn/hiddenlayer_0/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*'
_output_shapes
:���������Q
W
zero_fraction/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
zero_fraction/EqualEqual$dnn/hiddenlayer_0/hiddenlayer_0/Reluzero_fraction/zero*
T0*'
_output_shapes
:���������Q
p
zero_fraction/CastCastzero_fraction/Equal*

DstT0*

SrcT0
*'
_output_shapes
:���������Q
d
zero_fraction/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
zero_fraction/MeanMeanzero_fraction/Castzero_fraction/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
.dnn/hiddenlayer_0_fraction_of_zero_values/tagsConst*
dtype0*:
value1B/ B)dnn/hiddenlayer_0_fraction_of_zero_values*
_output_shapes
: 
�
)dnn/hiddenlayer_0_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_0_fraction_of_zero_values/tagszero_fraction/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_0_activation/tagConst*
dtype0*-
value$B" Bdnn/hiddenlayer_0_activation*
_output_shapes
: 
�
dnn/hiddenlayer_0_activationHistogramSummary dnn/hiddenlayer_0_activation/tag$dnn/hiddenlayer_0/hiddenlayer_0/Relu*
T0*
_output_shapes
: 
�
Adnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB"Q   Q   *
_output_shapes
:
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *�E�*
_output_shapes
: 
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *�E>*
_output_shapes
: 
�
Idnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:QQ*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes
: 
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:QQ
�
;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:QQ
�
 dnn/hiddenlayer_1/weights/part_0
VariableV2*
	container *
_output_shapes

:QQ*
dtype0*
shape
:QQ*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
shared_name 
�
'dnn/hiddenlayer_1/weights/part_0/AssignAssign dnn/hiddenlayer_1/weights/part_0;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking(*
T0*
_output_shapes

:QQ
�
%dnn/hiddenlayer_1/weights/part_0/readIdentity dnn/hiddenlayer_1/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:QQ
�
1dnn/hiddenlayer_1/biases/part_0/Initializer/ConstConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
valueBQ*    *
_output_shapes
:Q
�
dnn/hiddenlayer_1/biases/part_0
VariableV2*
	container *
_output_shapes
:Q*
dtype0*
shape:Q*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
shared_name 
�
&dnn/hiddenlayer_1/biases/part_0/AssignAssigndnn/hiddenlayer_1/biases/part_01dnn/hiddenlayer_1/biases/part_0/Initializer/Const*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q
�
$dnn/hiddenlayer_1/biases/part_0/readIdentitydnn/hiddenlayer_1/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
T0*
_output_shapes
:Q
u
dnn/hiddenlayer_1/weightsIdentity%dnn/hiddenlayer_1/weights/part_0/read*
T0*
_output_shapes

:QQ
�
dnn/hiddenlayer_1/MatMulMatMul$dnn/hiddenlayer_0/hiddenlayer_0/Reludnn/hiddenlayer_1/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������Q
o
dnn/hiddenlayer_1/biasesIdentity$dnn/hiddenlayer_1/biases/part_0/read*
T0*
_output_shapes
:Q
�
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/biases*
data_formatNHWC*
T0*'
_output_shapes
:���������Q
y
$dnn/hiddenlayer_1/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:���������Q
Y
zero_fraction_1/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
zero_fraction_1/EqualEqual$dnn/hiddenlayer_1/hiddenlayer_1/Reluzero_fraction_1/zero*
T0*'
_output_shapes
:���������Q
t
zero_fraction_1/CastCastzero_fraction_1/Equal*

DstT0*

SrcT0
*'
_output_shapes
:���������Q
f
zero_fraction_1/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
zero_fraction_1/MeanMeanzero_fraction_1/Castzero_fraction_1/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
.dnn/hiddenlayer_1_fraction_of_zero_values/tagsConst*
dtype0*:
value1B/ B)dnn/hiddenlayer_1_fraction_of_zero_values*
_output_shapes
: 
�
)dnn/hiddenlayer_1_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_1_fraction_of_zero_values/tagszero_fraction_1/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_1_activation/tagConst*
dtype0*-
value$B" Bdnn/hiddenlayer_1_activation*
_output_shapes
: 
�
dnn/hiddenlayer_1_activationHistogramSummary dnn/hiddenlayer_1_activation/tag$dnn/hiddenlayer_1/hiddenlayer_1/Relu*
T0*
_output_shapes
: 
�
Adnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB"Q   1   *
_output_shapes
:
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB
 *��[�*
_output_shapes
: 
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB
 *��[>*
_output_shapes
: 
�
Idnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:Q1*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes
: 
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:Q1
�
;dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:Q1
�
 dnn/hiddenlayer_2/weights/part_0
VariableV2*
	container *
_output_shapes

:Q1*
dtype0*
shape
:Q1*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
shared_name 
�
'dnn/hiddenlayer_2/weights/part_0/AssignAssign dnn/hiddenlayer_2/weights/part_0;dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
use_locking(*
T0*
_output_shapes

:Q1
�
%dnn/hiddenlayer_2/weights/part_0/readIdentity dnn/hiddenlayer_2/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:Q1
�
1dnn/hiddenlayer_2/biases/part_0/Initializer/ConstConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
valueB1*    *
_output_shapes
:1
�
dnn/hiddenlayer_2/biases/part_0
VariableV2*
	container *
_output_shapes
:1*
dtype0*
shape:1*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
shared_name 
�
&dnn/hiddenlayer_2/biases/part_0/AssignAssigndnn/hiddenlayer_2/biases/part_01dnn/hiddenlayer_2/biases/part_0/Initializer/Const*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
use_locking(*
T0*
_output_shapes
:1
�
$dnn/hiddenlayer_2/biases/part_0/readIdentitydnn/hiddenlayer_2/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
T0*
_output_shapes
:1
u
dnn/hiddenlayer_2/weightsIdentity%dnn/hiddenlayer_2/weights/part_0/read*
T0*
_output_shapes

:Q1
�
dnn/hiddenlayer_2/MatMulMatMul$dnn/hiddenlayer_1/hiddenlayer_1/Reludnn/hiddenlayer_2/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������1
o
dnn/hiddenlayer_2/biasesIdentity$dnn/hiddenlayer_2/biases/part_0/read*
T0*
_output_shapes
:1
�
dnn/hiddenlayer_2/BiasAddBiasAdddnn/hiddenlayer_2/MatMuldnn/hiddenlayer_2/biases*
data_formatNHWC*
T0*'
_output_shapes
:���������1
y
$dnn/hiddenlayer_2/hiddenlayer_2/ReluReludnn/hiddenlayer_2/BiasAdd*
T0*'
_output_shapes
:���������1
Y
zero_fraction_2/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
zero_fraction_2/EqualEqual$dnn/hiddenlayer_2/hiddenlayer_2/Reluzero_fraction_2/zero*
T0*'
_output_shapes
:���������1
t
zero_fraction_2/CastCastzero_fraction_2/Equal*

DstT0*

SrcT0
*'
_output_shapes
:���������1
f
zero_fraction_2/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
zero_fraction_2/MeanMeanzero_fraction_2/Castzero_fraction_2/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
.dnn/hiddenlayer_2_fraction_of_zero_values/tagsConst*
dtype0*:
value1B/ B)dnn/hiddenlayer_2_fraction_of_zero_values*
_output_shapes
: 
�
)dnn/hiddenlayer_2_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_2_fraction_of_zero_values/tagszero_fraction_2/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_2_activation/tagConst*
dtype0*-
value$B" Bdnn/hiddenlayer_2_activation*
_output_shapes
: 
�
dnn/hiddenlayer_2_activationHistogramSummary dnn/hiddenlayer_2_activation/tag$dnn/hiddenlayer_2/hiddenlayer_2/Relu*
T0*
_output_shapes
: 
�
Adnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
valueB"1      *
_output_shapes
:
�
?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
valueB
 *iʑ�*
_output_shapes
: 
�
?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
valueB
 *iʑ>*
_output_shapes
: 
�
Idnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:1*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0
�
?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
T0*
_output_shapes
: 
�
?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
T0*
_output_shapes

:1
�
;dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
T0*
_output_shapes

:1
�
 dnn/hiddenlayer_3/weights/part_0
VariableV2*
	container *
_output_shapes

:1*
dtype0*
shape
:1*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
shared_name 
�
'dnn/hiddenlayer_3/weights/part_0/AssignAssign dnn/hiddenlayer_3/weights/part_0;dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
use_locking(*
T0*
_output_shapes

:1
�
%dnn/hiddenlayer_3/weights/part_0/readIdentity dnn/hiddenlayer_3/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
T0*
_output_shapes

:1
�
1dnn/hiddenlayer_3/biases/part_0/Initializer/ConstConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
valueB*    *
_output_shapes
:
�
dnn/hiddenlayer_3/biases/part_0
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
shared_name 
�
&dnn/hiddenlayer_3/biases/part_0/AssignAssigndnn/hiddenlayer_3/biases/part_01dnn/hiddenlayer_3/biases/part_0/Initializer/Const*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
use_locking(*
T0*
_output_shapes
:
�
$dnn/hiddenlayer_3/biases/part_0/readIdentitydnn/hiddenlayer_3/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
T0*
_output_shapes
:
u
dnn/hiddenlayer_3/weightsIdentity%dnn/hiddenlayer_3/weights/part_0/read*
T0*
_output_shapes

:1
�
dnn/hiddenlayer_3/MatMulMatMul$dnn/hiddenlayer_2/hiddenlayer_2/Reludnn/hiddenlayer_3/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
o
dnn/hiddenlayer_3/biasesIdentity$dnn/hiddenlayer_3/biases/part_0/read*
T0*
_output_shapes
:
�
dnn/hiddenlayer_3/BiasAddBiasAdddnn/hiddenlayer_3/MatMuldnn/hiddenlayer_3/biases*
data_formatNHWC*
T0*'
_output_shapes
:���������
y
$dnn/hiddenlayer_3/hiddenlayer_3/ReluReludnn/hiddenlayer_3/BiasAdd*
T0*'
_output_shapes
:���������
Y
zero_fraction_3/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
zero_fraction_3/EqualEqual$dnn/hiddenlayer_3/hiddenlayer_3/Reluzero_fraction_3/zero*
T0*'
_output_shapes
:���������
t
zero_fraction_3/CastCastzero_fraction_3/Equal*

DstT0*

SrcT0
*'
_output_shapes
:���������
f
zero_fraction_3/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
zero_fraction_3/MeanMeanzero_fraction_3/Castzero_fraction_3/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
.dnn/hiddenlayer_3_fraction_of_zero_values/tagsConst*
dtype0*:
value1B/ B)dnn/hiddenlayer_3_fraction_of_zero_values*
_output_shapes
: 
�
)dnn/hiddenlayer_3_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_3_fraction_of_zero_values/tagszero_fraction_3/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_3_activation/tagConst*
dtype0*-
value$B" Bdnn/hiddenlayer_3_activation*
_output_shapes
: 
�
dnn/hiddenlayer_3_activationHistogramSummary dnn/hiddenlayer_3_activation/tag$dnn/hiddenlayer_3/hiddenlayer_3/Relu*
T0*
_output_shapes
: 
�
:dnn/logits/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB"      *
_output_shapes
:
�
8dnn/logits/weights/part_0/Initializer/random_uniform/minConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *����*
_output_shapes
: 
�
8dnn/logits/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *���>*
_output_shapes
: 
�
Bdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniform:dnn/logits/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:*
dtype0*
seed2 *

seed *
T0*,
_class"
 loc:@dnn/logits/weights/part_0
�
8dnn/logits/weights/part_0/Initializer/random_uniform/subSub8dnn/logits/weights/part_0/Initializer/random_uniform/max8dnn/logits/weights/part_0/Initializer/random_uniform/min*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes
: 
�
8dnn/logits/weights/part_0/Initializer/random_uniform/mulMulBdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniform8dnn/logits/weights/part_0/Initializer/random_uniform/sub*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

:
�
4dnn/logits/weights/part_0/Initializer/random_uniformAdd8dnn/logits/weights/part_0/Initializer/random_uniform/mul8dnn/logits/weights/part_0/Initializer/random_uniform/min*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

:
�
dnn/logits/weights/part_0
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*,
_class"
 loc:@dnn/logits/weights/part_0*
shared_name 
�
 dnn/logits/weights/part_0/AssignAssigndnn/logits/weights/part_04dnn/logits/weights/part_0/Initializer/random_uniform*
validate_shape(*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking(*
T0*
_output_shapes

:
�
dnn/logits/weights/part_0/readIdentitydnn/logits/weights/part_0*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

:
�
*dnn/logits/biases/part_0/Initializer/ConstConst*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
valueB*    *
_output_shapes
:
�
dnn/logits/biases/part_0
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*+
_class!
loc:@dnn/logits/biases/part_0*
shared_name 
�
dnn/logits/biases/part_0/AssignAssigndnn/logits/biases/part_0*dnn/logits/biases/part_0/Initializer/Const*
validate_shape(*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking(*
T0*
_output_shapes
:
�
dnn/logits/biases/part_0/readIdentitydnn/logits/biases/part_0*+
_class!
loc:@dnn/logits/biases/part_0*
T0*
_output_shapes
:
g
dnn/logits/weightsIdentitydnn/logits/weights/part_0/read*
T0*
_output_shapes

:
�
dnn/logits/MatMulMatMul$dnn/hiddenlayer_3/hiddenlayer_3/Reludnn/logits/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
a
dnn/logits/biasesIdentitydnn/logits/biases/part_0/read*
T0*
_output_shapes
:
�
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/biases*
data_formatNHWC*
T0*'
_output_shapes
:���������
Y
zero_fraction_4/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
z
zero_fraction_4/EqualEqualdnn/logits/BiasAddzero_fraction_4/zero*
T0*'
_output_shapes
:���������
t
zero_fraction_4/CastCastzero_fraction_4/Equal*

DstT0*

SrcT0
*'
_output_shapes
:���������
f
zero_fraction_4/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
zero_fraction_4/MeanMeanzero_fraction_4/Castzero_fraction_4/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
'dnn/logits_fraction_of_zero_values/tagsConst*
dtype0*3
value*B( B"dnn/logits_fraction_of_zero_values*
_output_shapes
: 
�
"dnn/logits_fraction_of_zero_valuesScalarSummary'dnn/logits_fraction_of_zero_values/tagszero_fraction_4/Mean*
T0*
_output_shapes
: 
o
dnn/logits_activation/tagConst*
dtype0*&
valueB Bdnn/logits_activation*
_output_shapes
: 
y
dnn/logits_activationHistogramSummarydnn/logits_activation/tagdnn/logits/BiasAdd*
T0*
_output_shapes
: 
v
predictions/scoresSqueezednn/logits/BiasAdd*
squeeze_dims
*
T0*#
_output_shapes
:���������
x
.training_loss/mean_squared_loss/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
�
*training_loss/mean_squared_loss/ExpandDims
ExpandDimsoutput.training_loss/mean_squared_loss/ExpandDims/dim*

Tdim0*
T0	*'
_output_shapes
:���������
�
'training_loss/mean_squared_loss/ToFloatCast*training_loss/mean_squared_loss/ExpandDims*

DstT0*

SrcT0	*'
_output_shapes
:���������
�
#training_loss/mean_squared_loss/subSubdnn/logits/BiasAdd'training_loss/mean_squared_loss/ToFloat*
T0*'
_output_shapes
:���������
�
training_loss/mean_squared_lossSquare#training_loss/mean_squared_loss/sub*
T0*'
_output_shapes
:���������
d
training_loss/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
training_lossMeantraining_loss/mean_squared_losstraining_loss/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
e
 training_loss/ScalarSummary/tagsConst*
dtype0*
valueB
 Bloss*
_output_shapes
: 
~
training_loss/ScalarSummaryScalarSummary training_loss/ScalarSummary/tagstraining_loss*
T0*
_output_shapes
: 
�
#dnn/learning_rate/Initializer/ConstConst*
dtype0*$
_class
loc:@dnn/learning_rate*
valueB
 *��L=*
_output_shapes
: 
�
dnn/learning_rate
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *$
_class
loc:@dnn/learning_rate*
shared_name 
�
dnn/learning_rate/AssignAssigndnn/learning_rate#dnn/learning_rate/Initializer/Const*
validate_shape(*$
_class
loc:@dnn/learning_rate*
use_locking(*
T0*
_output_shapes
: 
|
dnn/learning_rate/readIdentitydnn/learning_rate*$
_class
loc:@dnn/learning_rate*
T0*
_output_shapes
: 
_
train_op/dnn/gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
a
train_op/dnn/gradients/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
train_op/dnn/gradients/FillFilltrain_op/dnn/gradients/Shapetrain_op/dnn/gradients/Const*
T0*
_output_shapes
: 
�
7train_op/dnn/gradients/training_loss_grad/Reshape/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
�
1train_op/dnn/gradients/training_loss_grad/ReshapeReshapetrain_op/dnn/gradients/Fill7train_op/dnn/gradients/training_loss_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes

:
�
/train_op/dnn/gradients/training_loss_grad/ShapeShapetraining_loss/mean_squared_loss*
out_type0*
T0*
_output_shapes
:
�
.train_op/dnn/gradients/training_loss_grad/TileTile1train_op/dnn/gradients/training_loss_grad/Reshape/train_op/dnn/gradients/training_loss_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
�
1train_op/dnn/gradients/training_loss_grad/Shape_1Shapetraining_loss/mean_squared_loss*
out_type0*
T0*
_output_shapes
:
t
1train_op/dnn/gradients/training_loss_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
y
/train_op/dnn/gradients/training_loss_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
.train_op/dnn/gradients/training_loss_grad/ProdProd1train_op/dnn/gradients/training_loss_grad/Shape_1/train_op/dnn/gradients/training_loss_grad/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
{
1train_op/dnn/gradients/training_loss_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
�
0train_op/dnn/gradients/training_loss_grad/Prod_1Prod1train_op/dnn/gradients/training_loss_grad/Shape_21train_op/dnn/gradients/training_loss_grad/Const_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
u
3train_op/dnn/gradients/training_loss_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
1train_op/dnn/gradients/training_loss_grad/MaximumMaximum0train_op/dnn/gradients/training_loss_grad/Prod_13train_op/dnn/gradients/training_loss_grad/Maximum/y*
T0*
_output_shapes
: 
�
2train_op/dnn/gradients/training_loss_grad/floordivFloorDiv.train_op/dnn/gradients/training_loss_grad/Prod1train_op/dnn/gradients/training_loss_grad/Maximum*
T0*
_output_shapes
: 
�
.train_op/dnn/gradients/training_loss_grad/CastCast2train_op/dnn/gradients/training_loss_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 
�
1train_op/dnn/gradients/training_loss_grad/truedivRealDiv.train_op/dnn/gradients/training_loss_grad/Tile.train_op/dnn/gradients/training_loss_grad/Cast*
T0*'
_output_shapes
:���������
�
Atrain_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul/xConst2^train_op/dnn/gradients/training_loss_grad/truediv*
dtype0*
valueB
 *   @*
_output_shapes
: 
�
?train_op/dnn/gradients/training_loss/mean_squared_loss_grad/mulMulAtrain_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul/x#training_loss/mean_squared_loss/sub*
T0*'
_output_shapes
:���������
�
Atrain_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul_1Mul1train_op/dnn/gradients/training_loss_grad/truediv?train_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul*
T0*'
_output_shapes
:���������
�
Etrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/ShapeShapednn/logits/BiasAdd*
out_type0*
T0*
_output_shapes
:
�
Gtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Shape_1Shape'training_loss/mean_squared_loss/ToFloat*
out_type0*
T0*
_output_shapes
:
�
Utrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgsEtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/ShapeGtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ctrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/SumSumAtrain_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul_1Utrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
Gtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/ReshapeReshapeCtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/SumEtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
Etrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Sum_1SumAtrain_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul_1Wtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
Ctrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/NegNegEtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Sum_1*
T0*
_output_shapes
:
�
Itrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape_1ReshapeCtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/NegGtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:���������
�
Ptrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/group_depsNoOpH^train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/ReshapeJ^train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape_1
�
Xtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/control_dependencyIdentityGtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/ReshapeQ^train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/group_deps*Z
_classP
NLloc:@train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape*
T0*'
_output_shapes
:���������
�
Ztrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/control_dependency_1IdentityItrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape_1Q^train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/group_deps*\
_classR
PNloc:@train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
:train_op/dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGradBiasAddGradXtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/control_dependency*
data_formatNHWC*
T0*
_output_shapes
:
�
?train_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/group_depsNoOpY^train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/control_dependency;^train_op/dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGrad
�
Gtrain_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependencyIdentityXtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/control_dependency@^train_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/group_deps*Z
_classP
NLloc:@train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape*
T0*'
_output_shapes
:���������
�
Itrain_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependency_1Identity:train_op/dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGrad@^train_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/group_deps*M
_classC
A?loc:@train_op/dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
�
4train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMulMatMulGtrain_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependencydnn/logits/weights*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������
�
6train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul_1MatMul$dnn/hiddenlayer_3/hiddenlayer_3/ReluGtrain_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:
�
>train_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/group_depsNoOp5^train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul7^train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul_1
�
Ftrain_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependencyIdentity4train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul?^train_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/group_deps*G
_class=
;9loc:@train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������
�
Htrain_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependency_1Identity6train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul_1?^train_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/group_deps*I
_class?
=;loc:@train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
�
Itrain_op/dnn/gradients/dnn/hiddenlayer_3/hiddenlayer_3/Relu_grad/ReluGradReluGradFtrain_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependency$dnn/hiddenlayer_3/hiddenlayer_3/Relu*
T0*'
_output_shapes
:���������
�
Atrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/BiasAddGradBiasAddGradItrain_op/dnn/gradients/dnn/hiddenlayer_3/hiddenlayer_3/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:
�
Ftrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/group_depsNoOpJ^train_op/dnn/gradients/dnn/hiddenlayer_3/hiddenlayer_3/Relu_grad/ReluGradB^train_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/BiasAddGrad
�
Ntrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/control_dependencyIdentityItrain_op/dnn/gradients/dnn/hiddenlayer_3/hiddenlayer_3/Relu_grad/ReluGradG^train_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@train_op/dnn/gradients/dnn/hiddenlayer_3/hiddenlayer_3/Relu_grad/ReluGrad*
T0*'
_output_shapes
:���������
�
Ptrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/control_dependency_1IdentityAtrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/BiasAddGradG^train_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/group_deps*T
_classJ
HFloc:@train_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
�
;train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMulMatMulNtrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/control_dependencydnn/hiddenlayer_3/weights*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������1
�
=train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul_1MatMul$dnn/hiddenlayer_2/hiddenlayer_2/ReluNtrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:1
�
Etrain_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/group_depsNoOp<^train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul>^train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul_1
�
Mtrain_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/control_dependencyIdentity;train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMulF^train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������1
�
Otrain_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/control_dependency_1Identity=train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul_1F^train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

:1
�
Itrain_op/dnn/gradients/dnn/hiddenlayer_2/hiddenlayer_2/Relu_grad/ReluGradReluGradMtrain_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/control_dependency$dnn/hiddenlayer_2/hiddenlayer_2/Relu*
T0*'
_output_shapes
:���������1
�
Atrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/BiasAddGradBiasAddGradItrain_op/dnn/gradients/dnn/hiddenlayer_2/hiddenlayer_2/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:1
�
Ftrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/group_depsNoOpJ^train_op/dnn/gradients/dnn/hiddenlayer_2/hiddenlayer_2/Relu_grad/ReluGradB^train_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/BiasAddGrad
�
Ntrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/control_dependencyIdentityItrain_op/dnn/gradients/dnn/hiddenlayer_2/hiddenlayer_2/Relu_grad/ReluGradG^train_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@train_op/dnn/gradients/dnn/hiddenlayer_2/hiddenlayer_2/Relu_grad/ReluGrad*
T0*'
_output_shapes
:���������1
�
Ptrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/control_dependency_1IdentityAtrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/BiasAddGradG^train_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/group_deps*T
_classJ
HFloc:@train_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:1
�
;train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMulMatMulNtrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/control_dependencydnn/hiddenlayer_2/weights*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������Q
�
=train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul_1MatMul$dnn/hiddenlayer_1/hiddenlayer_1/ReluNtrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:Q1
�
Etrain_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/group_depsNoOp<^train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul>^train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul_1
�
Mtrain_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/control_dependencyIdentity;train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMulF^train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������Q
�
Otrain_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/control_dependency_1Identity=train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul_1F^train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:Q1
�
Itrain_op/dnn/gradients/dnn/hiddenlayer_1/hiddenlayer_1/Relu_grad/ReluGradReluGradMtrain_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/control_dependency$dnn/hiddenlayer_1/hiddenlayer_1/Relu*
T0*'
_output_shapes
:���������Q
�
Atrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGradBiasAddGradItrain_op/dnn/gradients/dnn/hiddenlayer_1/hiddenlayer_1/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:Q
�
Ftrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/group_depsNoOpJ^train_op/dnn/gradients/dnn/hiddenlayer_1/hiddenlayer_1/Relu_grad/ReluGradB^train_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGrad
�
Ntrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependencyIdentityItrain_op/dnn/gradients/dnn/hiddenlayer_1/hiddenlayer_1/Relu_grad/ReluGradG^train_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@train_op/dnn/gradients/dnn/hiddenlayer_1/hiddenlayer_1/Relu_grad/ReluGrad*
T0*'
_output_shapes
:���������Q
�
Ptrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependency_1IdentityAtrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGradG^train_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/group_deps*T
_classJ
HFloc:@train_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:Q
�
;train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMulMatMulNtrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependencydnn/hiddenlayer_1/weights*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������Q
�
=train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1MatMul$dnn/hiddenlayer_0/hiddenlayer_0/ReluNtrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:QQ
�
Etrain_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/group_depsNoOp<^train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul>^train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1
�
Mtrain_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependencyIdentity;train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMulF^train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������Q
�
Otrain_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependency_1Identity=train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1F^train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:QQ
�
Itrain_op/dnn/gradients/dnn/hiddenlayer_0/hiddenlayer_0/Relu_grad/ReluGradReluGradMtrain_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependency$dnn/hiddenlayer_0/hiddenlayer_0/Relu*
T0*'
_output_shapes
:���������Q
�
Atrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGradBiasAddGradItrain_op/dnn/gradients/dnn/hiddenlayer_0/hiddenlayer_0/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:Q
�
Ftrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/group_depsNoOpJ^train_op/dnn/gradients/dnn/hiddenlayer_0/hiddenlayer_0/Relu_grad/ReluGradB^train_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGrad
�
Ntrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependencyIdentityItrain_op/dnn/gradients/dnn/hiddenlayer_0/hiddenlayer_0/Relu_grad/ReluGradG^train_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@train_op/dnn/gradients/dnn/hiddenlayer_0/hiddenlayer_0/Relu_grad/ReluGrad*
T0*'
_output_shapes
:���������Q
�
Ptrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependency_1IdentityAtrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGradG^train_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/group_deps*T
_classJ
HFloc:@train_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:Q
�
;train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMulMatMulNtrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependencydnn/hiddenlayer_0/weights*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������T
�
=train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1MatMul@dnn/input_from_feature_columns/input_from_feature_columns/concatNtrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:TQ
�
Etrain_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/group_depsNoOp<^train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul>^train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1
�
Mtrain_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/control_dependencyIdentity;train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMulF^train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������T
�
Otrain_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/control_dependency_1Identity=train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1F^train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1*
T0*
_output_shapes

:TQ
�
train_op/dnn/ConstConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueBTQ*���=*
_output_shapes

:TQ
�
,dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad
VariableV2*
	container *
_output_shapes

:TQ*
dtype0*
shape
:TQ*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
shared_name 
�
3dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad/AssignAssign,dnn/dnn/hiddenlayer_0/weights/part_0/Adagradtrain_op/dnn/Const*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking(*
T0*
_output_shapes

:TQ
�
1dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad/readIdentity,dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:TQ
�
train_op/dnn/Const_1Const*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
valueBQ*���=*
_output_shapes
:Q
�
+dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad
VariableV2*
	container *
_output_shapes
:Q*
dtype0*
shape:Q*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
shared_name 
�
2dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad/AssignAssign+dnn/dnn/hiddenlayer_0/biases/part_0/Adagradtrain_op/dnn/Const_1*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q
�
0dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad/readIdentity+dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
T0*
_output_shapes
:Q
�
train_op/dnn/Const_2Const*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueBQQ*���=*
_output_shapes

:QQ
�
,dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad
VariableV2*
	container *
_output_shapes

:QQ*
dtype0*
shape
:QQ*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
shared_name 
�
3dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad/AssignAssign,dnn/dnn/hiddenlayer_1/weights/part_0/Adagradtrain_op/dnn/Const_2*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking(*
T0*
_output_shapes

:QQ
�
1dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad/readIdentity,dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:QQ
�
train_op/dnn/Const_3Const*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
valueBQ*���=*
_output_shapes
:Q
�
+dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad
VariableV2*
	container *
_output_shapes
:Q*
dtype0*
shape:Q*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
shared_name 
�
2dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad/AssignAssign+dnn/dnn/hiddenlayer_1/biases/part_0/Adagradtrain_op/dnn/Const_3*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q
�
0dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad/readIdentity+dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
T0*
_output_shapes
:Q
�
train_op/dnn/Const_4Const*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueBQ1*���=*
_output_shapes

:Q1
�
,dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad
VariableV2*
	container *
_output_shapes

:Q1*
dtype0*
shape
:Q1*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
shared_name 
�
3dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad/AssignAssign,dnn/dnn/hiddenlayer_2/weights/part_0/Adagradtrain_op/dnn/Const_4*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
use_locking(*
T0*
_output_shapes

:Q1
�
1dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad/readIdentity,dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:Q1
�
train_op/dnn/Const_5Const*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
valueB1*���=*
_output_shapes
:1
�
+dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad
VariableV2*
	container *
_output_shapes
:1*
dtype0*
shape:1*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
shared_name 
�
2dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad/AssignAssign+dnn/dnn/hiddenlayer_2/biases/part_0/Adagradtrain_op/dnn/Const_5*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
use_locking(*
T0*
_output_shapes
:1
�
0dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad/readIdentity+dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
T0*
_output_shapes
:1
�
train_op/dnn/Const_6Const*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
valueB1*���=*
_output_shapes

:1
�
,dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad
VariableV2*
	container *
_output_shapes

:1*
dtype0*
shape
:1*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
shared_name 
�
3dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad/AssignAssign,dnn/dnn/hiddenlayer_3/weights/part_0/Adagradtrain_op/dnn/Const_6*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
use_locking(*
T0*
_output_shapes

:1
�
1dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad/readIdentity,dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
T0*
_output_shapes

:1
�
train_op/dnn/Const_7Const*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
valueB*���=*
_output_shapes
:
�
+dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
shared_name 
�
2dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad/AssignAssign+dnn/dnn/hiddenlayer_3/biases/part_0/Adagradtrain_op/dnn/Const_7*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
use_locking(*
T0*
_output_shapes
:
�
0dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad/readIdentity+dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
T0*
_output_shapes
:
�
train_op/dnn/Const_8Const*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB*���=*
_output_shapes

:
�
%dnn/dnn/logits/weights/part_0/Adagrad
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*,
_class"
 loc:@dnn/logits/weights/part_0*
shared_name 
�
,dnn/dnn/logits/weights/part_0/Adagrad/AssignAssign%dnn/dnn/logits/weights/part_0/Adagradtrain_op/dnn/Const_8*
validate_shape(*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking(*
T0*
_output_shapes

:
�
*dnn/dnn/logits/weights/part_0/Adagrad/readIdentity%dnn/dnn/logits/weights/part_0/Adagrad*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

:
�
train_op/dnn/Const_9Const*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
valueB*���=*
_output_shapes
:
�
$dnn/dnn/logits/biases/part_0/Adagrad
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*+
_class!
loc:@dnn/logits/biases/part_0*
shared_name 
�
+dnn/dnn/logits/biases/part_0/Adagrad/AssignAssign$dnn/dnn/logits/biases/part_0/Adagradtrain_op/dnn/Const_9*
validate_shape(*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking(*
T0*
_output_shapes
:
�
)dnn/dnn/logits/biases/part_0/Adagrad/readIdentity$dnn/dnn/logits/biases/part_0/Adagrad*+
_class!
loc:@dnn/logits/biases/part_0*
T0*
_output_shapes
:
�
Gtrain_op/dnn/train/update_dnn/hiddenlayer_0/weights/part_0/ApplyAdagradApplyAdagrad dnn/hiddenlayer_0/weights/part_0,dnn/dnn/hiddenlayer_0/weights/part_0/Adagraddnn/learning_rate/readOtrain_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/control_dependency_1*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking( *
T0*
_output_shapes

:TQ
�
Ftrain_op/dnn/train/update_dnn/hiddenlayer_0/biases/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_0/biases/part_0+dnn/dnn/hiddenlayer_0/biases/part_0/Adagraddnn/learning_rate/readPtrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependency_1*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking( *
T0*
_output_shapes
:Q
�
Gtrain_op/dnn/train/update_dnn/hiddenlayer_1/weights/part_0/ApplyAdagradApplyAdagrad dnn/hiddenlayer_1/weights/part_0,dnn/dnn/hiddenlayer_1/weights/part_0/Adagraddnn/learning_rate/readOtrain_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependency_1*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking( *
T0*
_output_shapes

:QQ
�
Ftrain_op/dnn/train/update_dnn/hiddenlayer_1/biases/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_1/biases/part_0+dnn/dnn/hiddenlayer_1/biases/part_0/Adagraddnn/learning_rate/readPtrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependency_1*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking( *
T0*
_output_shapes
:Q
�
Gtrain_op/dnn/train/update_dnn/hiddenlayer_2/weights/part_0/ApplyAdagradApplyAdagrad dnn/hiddenlayer_2/weights/part_0,dnn/dnn/hiddenlayer_2/weights/part_0/Adagraddnn/learning_rate/readOtrain_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/control_dependency_1*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
use_locking( *
T0*
_output_shapes

:Q1
�
Ftrain_op/dnn/train/update_dnn/hiddenlayer_2/biases/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_2/biases/part_0+dnn/dnn/hiddenlayer_2/biases/part_0/Adagraddnn/learning_rate/readPtrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/control_dependency_1*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
use_locking( *
T0*
_output_shapes
:1
�
Gtrain_op/dnn/train/update_dnn/hiddenlayer_3/weights/part_0/ApplyAdagradApplyAdagrad dnn/hiddenlayer_3/weights/part_0,dnn/dnn/hiddenlayer_3/weights/part_0/Adagraddnn/learning_rate/readOtrain_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/control_dependency_1*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
use_locking( *
T0*
_output_shapes

:1
�
Ftrain_op/dnn/train/update_dnn/hiddenlayer_3/biases/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_3/biases/part_0+dnn/dnn/hiddenlayer_3/biases/part_0/Adagraddnn/learning_rate/readPtrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/control_dependency_1*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
use_locking( *
T0*
_output_shapes
:
�
@train_op/dnn/train/update_dnn/logits/weights/part_0/ApplyAdagradApplyAdagraddnn/logits/weights/part_0%dnn/dnn/logits/weights/part_0/Adagraddnn/learning_rate/readHtrain_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependency_1*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking( *
T0*
_output_shapes

:
�
?train_op/dnn/train/update_dnn/logits/biases/part_0/ApplyAdagradApplyAdagraddnn/logits/biases/part_0$dnn/dnn/logits/biases/part_0/Adagraddnn/learning_rate/readItrain_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependency_1*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking( *
T0*
_output_shapes
:
�
train_op/dnn/train/updateNoOpH^train_op/dnn/train/update_dnn/hiddenlayer_0/weights/part_0/ApplyAdagradG^train_op/dnn/train/update_dnn/hiddenlayer_0/biases/part_0/ApplyAdagradH^train_op/dnn/train/update_dnn/hiddenlayer_1/weights/part_0/ApplyAdagradG^train_op/dnn/train/update_dnn/hiddenlayer_1/biases/part_0/ApplyAdagradH^train_op/dnn/train/update_dnn/hiddenlayer_2/weights/part_0/ApplyAdagradG^train_op/dnn/train/update_dnn/hiddenlayer_2/biases/part_0/ApplyAdagradH^train_op/dnn/train/update_dnn/hiddenlayer_3/weights/part_0/ApplyAdagradG^train_op/dnn/train/update_dnn/hiddenlayer_3/biases/part_0/ApplyAdagradA^train_op/dnn/train/update_dnn/logits/weights/part_0/ApplyAdagrad@^train_op/dnn/train/update_dnn/logits/biases/part_0/ApplyAdagrad
�
train_op/dnn/train/valueConst^train_op/dnn/train/update*
dtype0	*
_class
loc:@global_step*
value	B	 R*
_output_shapes
: 
�
train_op/dnn/train	AssignAddglobal_steptrain_op/dnn/train/value*
_class
loc:@global_step*
use_locking( *
T0	*
_output_shapes
: 
�
train_op/dnn/control_dependencyIdentitytraining_loss^train_op/dnn/train* 
_class
loc:@training_loss*
T0*
_output_shapes
: 
r
(metrics/mean_squared_loss/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
�
$metrics/mean_squared_loss/ExpandDims
ExpandDimsoutput(metrics/mean_squared_loss/ExpandDims/dim*

Tdim0*
T0	*'
_output_shapes
:���������
t
*metrics/mean_squared_loss/ExpandDims_1/dimConst*
dtype0*
valueB:*
_output_shapes
:
�
&metrics/mean_squared_loss/ExpandDims_1
ExpandDimspredictions/scores*metrics/mean_squared_loss/ExpandDims_1/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
!metrics/mean_squared_loss/ToFloatCast$metrics/mean_squared_loss/ExpandDims*

DstT0*

SrcT0	*'
_output_shapes
:���������
�
metrics/mean_squared_loss/subSub&metrics/mean_squared_loss/ExpandDims_1!metrics/mean_squared_loss/ToFloat*
T0*'
_output_shapes
:���������
t
metrics/mean_squared_lossSquaremetrics/mean_squared_loss/sub*
T0*'
_output_shapes
:���������
h
metrics/eval_loss/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
metrics/eval_lossMeanmetrics/mean_squared_lossmetrics/eval_loss/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
W
metrics/mean/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/total
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
�
metrics/mean/total/AssignAssignmetrics/mean/totalmetrics/mean/zeros*
validate_shape(*%
_class
loc:@metrics/mean/total*
use_locking(*
T0*
_output_shapes
: 

metrics/mean/total/readIdentitymetrics/mean/total*%
_class
loc:@metrics/mean/total*
T0*
_output_shapes
: 
Y
metrics/mean/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/count
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
�
metrics/mean/count/AssignAssignmetrics/mean/countmetrics/mean/zeros_1*
validate_shape(*%
_class
loc:@metrics/mean/count*
use_locking(*
T0*
_output_shapes
: 

metrics/mean/count/readIdentitymetrics/mean/count*%
_class
loc:@metrics/mean/count*
T0*
_output_shapes
: 
S
metrics/mean/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
a
metrics/mean/ToFloat_1Castmetrics/mean/Size*

DstT0*

SrcT0*
_output_shapes
: 
U
metrics/mean/ConstConst*
dtype0*
valueB *
_output_shapes
: 
|
metrics/mean/SumSummetrics/eval_lossmetrics/mean/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
metrics/mean/AssignAdd	AssignAddmetrics/mean/totalmetrics/mean/Sum*%
_class
loc:@metrics/mean/total*
use_locking( *
T0*
_output_shapes
: 
�
metrics/mean/AssignAdd_1	AssignAddmetrics/mean/countmetrics/mean/ToFloat_1*%
_class
loc:@metrics/mean/count*
use_locking( *
T0*
_output_shapes
: 
[
metrics/mean/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
q
metrics/mean/GreaterGreatermetrics/mean/count/readmetrics/mean/Greater/y*
T0*
_output_shapes
: 
r
metrics/mean/truedivRealDivmetrics/mean/total/readmetrics/mean/count/read*
T0*
_output_shapes
: 
Y
metrics/mean/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 

metrics/mean/valueSelectmetrics/mean/Greatermetrics/mean/truedivmetrics/mean/value/e*
T0*
_output_shapes
: 
]
metrics/mean/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/Greater_1Greatermetrics/mean/AssignAdd_1metrics/mean/Greater_1/y*
T0*
_output_shapes
: 
t
metrics/mean/truediv_1RealDivmetrics/mean/AssignAddmetrics/mean/AssignAdd_1*
T0*
_output_shapes
: 
]
metrics/mean/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
metrics/mean/update_opSelectmetrics/mean/Greater_1metrics/mean/truediv_1metrics/mean/update_op/e*
T0*
_output_shapes
: ""
losses

training_loss:0" 
global_step

global_step:0"�
trainable_variables��
�
"dnn/hiddenlayer_0/weights/part_0:0'dnn/hiddenlayer_0/weights/part_0/Assign'dnn/hiddenlayer_0/weights/part_0/read:0"'
dnn/hiddenlayer_0/weightsTQ  "TQ
�
!dnn/hiddenlayer_0/biases/part_0:0&dnn/hiddenlayer_0/biases/part_0/Assign&dnn/hiddenlayer_0/biases/part_0/read:0"#
dnn/hiddenlayer_0/biasesQ "Q
�
"dnn/hiddenlayer_1/weights/part_0:0'dnn/hiddenlayer_1/weights/part_0/Assign'dnn/hiddenlayer_1/weights/part_0/read:0"'
dnn/hiddenlayer_1/weightsQQ  "QQ
�
!dnn/hiddenlayer_1/biases/part_0:0&dnn/hiddenlayer_1/biases/part_0/Assign&dnn/hiddenlayer_1/biases/part_0/read:0"#
dnn/hiddenlayer_1/biasesQ "Q
�
"dnn/hiddenlayer_2/weights/part_0:0'dnn/hiddenlayer_2/weights/part_0/Assign'dnn/hiddenlayer_2/weights/part_0/read:0"'
dnn/hiddenlayer_2/weightsQ1  "Q1
�
!dnn/hiddenlayer_2/biases/part_0:0&dnn/hiddenlayer_2/biases/part_0/Assign&dnn/hiddenlayer_2/biases/part_0/read:0"#
dnn/hiddenlayer_2/biases1 "1
�
"dnn/hiddenlayer_3/weights/part_0:0'dnn/hiddenlayer_3/weights/part_0/Assign'dnn/hiddenlayer_3/weights/part_0/read:0"'
dnn/hiddenlayer_3/weights1  "1
�
!dnn/hiddenlayer_3/biases/part_0:0&dnn/hiddenlayer_3/biases/part_0/Assign&dnn/hiddenlayer_3/biases/part_0/read:0"#
dnn/hiddenlayer_3/biases "
�
dnn/logits/weights/part_0:0 dnn/logits/weights/part_0/Assign dnn/logits/weights/part_0/read:0" 
dnn/logits/weights  "
|
dnn/logits/biases/part_0:0dnn/logits/biases/part_0/Assigndnn/logits/biases/part_0/read:0"
dnn/logits/biases ""�
	variables��
7
global_step:0global_step/Assignglobal_step/read:0
�
"dnn/hiddenlayer_0/weights/part_0:0'dnn/hiddenlayer_0/weights/part_0/Assign'dnn/hiddenlayer_0/weights/part_0/read:0"'
dnn/hiddenlayer_0/weightsTQ  "TQ
�
!dnn/hiddenlayer_0/biases/part_0:0&dnn/hiddenlayer_0/biases/part_0/Assign&dnn/hiddenlayer_0/biases/part_0/read:0"#
dnn/hiddenlayer_0/biasesQ "Q
�
"dnn/hiddenlayer_1/weights/part_0:0'dnn/hiddenlayer_1/weights/part_0/Assign'dnn/hiddenlayer_1/weights/part_0/read:0"'
dnn/hiddenlayer_1/weightsQQ  "QQ
�
!dnn/hiddenlayer_1/biases/part_0:0&dnn/hiddenlayer_1/biases/part_0/Assign&dnn/hiddenlayer_1/biases/part_0/read:0"#
dnn/hiddenlayer_1/biasesQ "Q
�
"dnn/hiddenlayer_2/weights/part_0:0'dnn/hiddenlayer_2/weights/part_0/Assign'dnn/hiddenlayer_2/weights/part_0/read:0"'
dnn/hiddenlayer_2/weightsQ1  "Q1
�
!dnn/hiddenlayer_2/biases/part_0:0&dnn/hiddenlayer_2/biases/part_0/Assign&dnn/hiddenlayer_2/biases/part_0/read:0"#
dnn/hiddenlayer_2/biases1 "1
�
"dnn/hiddenlayer_3/weights/part_0:0'dnn/hiddenlayer_3/weights/part_0/Assign'dnn/hiddenlayer_3/weights/part_0/read:0"'
dnn/hiddenlayer_3/weights1  "1
�
!dnn/hiddenlayer_3/biases/part_0:0&dnn/hiddenlayer_3/biases/part_0/Assign&dnn/hiddenlayer_3/biases/part_0/read:0"#
dnn/hiddenlayer_3/biases "
�
dnn/logits/weights/part_0:0 dnn/logits/weights/part_0/Assign dnn/logits/weights/part_0/read:0" 
dnn/logits/weights  "
|
dnn/logits/biases/part_0:0dnn/logits/biases/part_0/Assigndnn/logits/biases/part_0/read:0"
dnn/logits/biases "
I
dnn/learning_rate:0dnn/learning_rate/Assigndnn/learning_rate/read:0
�
.dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad:03dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad/Assign3dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad/read:0"3
%dnn/hiddenlayer_0/weights/t_0/AdagradTQ  "TQ
�
-dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad:02dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad/Assign2dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad/read:0"/
$dnn/hiddenlayer_0/biases/t_0/AdagradQ "Q
�
.dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad:03dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad/Assign3dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad/read:0"3
%dnn/hiddenlayer_1/weights/t_0/AdagradQQ  "QQ
�
-dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad:02dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad/Assign2dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad/read:0"/
$dnn/hiddenlayer_1/biases/t_0/AdagradQ "Q
�
.dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad:03dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad/Assign3dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad/read:0"3
%dnn/hiddenlayer_2/weights/t_0/AdagradQ1  "Q1
�
-dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad:02dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad/Assign2dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad/read:0"/
$dnn/hiddenlayer_2/biases/t_0/Adagrad1 "1
�
.dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad:03dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad/Assign3dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad/read:0"3
%dnn/hiddenlayer_3/weights/t_0/Adagrad1  "1
�
-dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad:02dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad/Assign2dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad/read:0"/
$dnn/hiddenlayer_3/biases/t_0/Adagrad "
�
'dnn/dnn/logits/weights/part_0/Adagrad:0,dnn/dnn/logits/weights/part_0/Adagrad/Assign,dnn/dnn/logits/weights/part_0/Adagrad/read:0",
dnn/logits/weights/t_0/Adagrad  "
�
&dnn/dnn/logits/biases/part_0/Adagrad:0+dnn/dnn/logits/biases/part_0/Adagrad/Assign+dnn/dnn/logits/biases/part_0/Adagrad/read:0"(
dnn/logits/biases/t_0/Adagrad ""�
dnn�
�
"dnn/hiddenlayer_0/weights/part_0:0
!dnn/hiddenlayer_0/biases/part_0:0
"dnn/hiddenlayer_1/weights/part_0:0
!dnn/hiddenlayer_1/biases/part_0:0
"dnn/hiddenlayer_2/weights/part_0:0
!dnn/hiddenlayer_2/biases/part_0:0
"dnn/hiddenlayer_3/weights/part_0:0
!dnn/hiddenlayer_3/biases/part_0:0
dnn/logits/weights/part_0:0
dnn/logits/biases/part_0:0""
train_op

train_op/dnn/train"A
local_variables.
,
metrics/mean/total:0
metrics/mean/count:0"

savers "�
	summaries�
�
+dnn/hiddenlayer_0_fraction_of_zero_values:0
dnn/hiddenlayer_0_activation:0
+dnn/hiddenlayer_1_fraction_of_zero_values:0
dnn/hiddenlayer_1_activation:0
+dnn/hiddenlayer_2_fraction_of_zero_values:0
dnn/hiddenlayer_2_activation:0
+dnn/hiddenlayer_3_fraction_of_zero_values:0
dnn/hiddenlayer_3_activation:0
$dnn/logits_fraction_of_zero_values:0
dnn/logits_activation:0
training_loss/ScalarSummary:0"�
model_variables�
�
"dnn/hiddenlayer_0/weights/part_0:0
!dnn/hiddenlayer_0/biases/part_0:0
"dnn/hiddenlayer_1/weights/part_0:0
!dnn/hiddenlayer_1/biases/part_0:0
"dnn/hiddenlayer_2/weights/part_0:0
!dnn/hiddenlayer_2/biases/part_0:0
"dnn/hiddenlayer_3/weights/part_0:0
!dnn/hiddenlayer_3/biases/part_0:0
dnn/logits/weights/part_0:0
dnn/logits/biases/part_0:0�7�i��     Y��	�%�1�A"��

global_step/Initializer/ConstConst*
dtype0	*
_class
loc:@global_step*
value	B	 R *
_output_shapes
: 
�
global_step
VariableV2*
	container *
_output_shapes
: *
dtype0	*
shape: *
_class
loc:@global_step*
shared_name 
�
global_step/AssignAssignglobal_stepglobal_step/Initializer/Const*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
T0	*
_output_shapes
: 
W
inputPlaceholder*
dtype0*
shape: *'
_output_shapes
:���������T
T
outputPlaceholder*
dtype0	*
shape: *#
_output_shapes
:���������
�
Kdnn/input_from_feature_columns/input_from_feature_columns/concat/concat_dimConst*
dtype0*
value	B :*
_output_shapes
: 
�
@dnn/input_from_feature_columns/input_from_feature_columns/concatIdentityinput*
T0*'
_output_shapes
:���������T
�
Adnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB"T   Q   *
_output_shapes
:
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *�DC�*
_output_shapes
: 
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *�DC>*
_output_shapes
: 
�
Idnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:TQ*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes
: 
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:TQ
�
;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:TQ
�
 dnn/hiddenlayer_0/weights/part_0
VariableV2*
	container *
_output_shapes

:TQ*
dtype0*
shape
:TQ*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
shared_name 
�
'dnn/hiddenlayer_0/weights/part_0/AssignAssign dnn/hiddenlayer_0/weights/part_0;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking(*
T0*
_output_shapes

:TQ
�
%dnn/hiddenlayer_0/weights/part_0/readIdentity dnn/hiddenlayer_0/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:TQ
�
1dnn/hiddenlayer_0/biases/part_0/Initializer/ConstConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
valueBQ*    *
_output_shapes
:Q
�
dnn/hiddenlayer_0/biases/part_0
VariableV2*
	container *
_output_shapes
:Q*
dtype0*
shape:Q*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
shared_name 
�
&dnn/hiddenlayer_0/biases/part_0/AssignAssigndnn/hiddenlayer_0/biases/part_01dnn/hiddenlayer_0/biases/part_0/Initializer/Const*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q
�
$dnn/hiddenlayer_0/biases/part_0/readIdentitydnn/hiddenlayer_0/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
T0*
_output_shapes
:Q
u
dnn/hiddenlayer_0/weightsIdentity%dnn/hiddenlayer_0/weights/part_0/read*
T0*
_output_shapes

:TQ
�
dnn/hiddenlayer_0/MatMulMatMul@dnn/input_from_feature_columns/input_from_feature_columns/concatdnn/hiddenlayer_0/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������Q
o
dnn/hiddenlayer_0/biasesIdentity$dnn/hiddenlayer_0/biases/part_0/read*
T0*
_output_shapes
:Q
�
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/biases*'
_output_shapes
:���������Q*
T0*
data_formatNHWC
y
$dnn/hiddenlayer_0/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*'
_output_shapes
:���������Q
W
zero_fraction/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
zero_fraction/EqualEqual$dnn/hiddenlayer_0/hiddenlayer_0/Reluzero_fraction/zero*
T0*'
_output_shapes
:���������Q
p
zero_fraction/CastCastzero_fraction/Equal*

DstT0*

SrcT0
*'
_output_shapes
:���������Q
d
zero_fraction/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
zero_fraction/MeanMeanzero_fraction/Castzero_fraction/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
.dnn/hiddenlayer_0_fraction_of_zero_values/tagsConst*
dtype0*:
value1B/ B)dnn/hiddenlayer_0_fraction_of_zero_values*
_output_shapes
: 
�
)dnn/hiddenlayer_0_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_0_fraction_of_zero_values/tagszero_fraction/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_0_activation/tagConst*
dtype0*-
value$B" Bdnn/hiddenlayer_0_activation*
_output_shapes
: 
�
dnn/hiddenlayer_0_activationHistogramSummary dnn/hiddenlayer_0_activation/tag$dnn/hiddenlayer_0/hiddenlayer_0/Relu*
T0*
_output_shapes
: 
�
Adnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB"Q   Q   *
_output_shapes
:
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *�E�*
_output_shapes
: 
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *�E>*
_output_shapes
: 
�
Idnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:QQ*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes
: 
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:QQ
�
;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:QQ
�
 dnn/hiddenlayer_1/weights/part_0
VariableV2*
	container *
_output_shapes

:QQ*
dtype0*
shape
:QQ*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
shared_name 
�
'dnn/hiddenlayer_1/weights/part_0/AssignAssign dnn/hiddenlayer_1/weights/part_0;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking(*
T0*
_output_shapes

:QQ
�
%dnn/hiddenlayer_1/weights/part_0/readIdentity dnn/hiddenlayer_1/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:QQ
�
1dnn/hiddenlayer_1/biases/part_0/Initializer/ConstConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
valueBQ*    *
_output_shapes
:Q
�
dnn/hiddenlayer_1/biases/part_0
VariableV2*
	container *
_output_shapes
:Q*
dtype0*
shape:Q*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
shared_name 
�
&dnn/hiddenlayer_1/biases/part_0/AssignAssigndnn/hiddenlayer_1/biases/part_01dnn/hiddenlayer_1/biases/part_0/Initializer/Const*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q
�
$dnn/hiddenlayer_1/biases/part_0/readIdentitydnn/hiddenlayer_1/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
T0*
_output_shapes
:Q
u
dnn/hiddenlayer_1/weightsIdentity%dnn/hiddenlayer_1/weights/part_0/read*
T0*
_output_shapes

:QQ
�
dnn/hiddenlayer_1/MatMulMatMul$dnn/hiddenlayer_0/hiddenlayer_0/Reludnn/hiddenlayer_1/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������Q
o
dnn/hiddenlayer_1/biasesIdentity$dnn/hiddenlayer_1/biases/part_0/read*
T0*
_output_shapes
:Q
�
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/biases*'
_output_shapes
:���������Q*
T0*
data_formatNHWC
y
$dnn/hiddenlayer_1/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:���������Q
Y
zero_fraction_1/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
zero_fraction_1/EqualEqual$dnn/hiddenlayer_1/hiddenlayer_1/Reluzero_fraction_1/zero*
T0*'
_output_shapes
:���������Q
t
zero_fraction_1/CastCastzero_fraction_1/Equal*

DstT0*

SrcT0
*'
_output_shapes
:���������Q
f
zero_fraction_1/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
zero_fraction_1/MeanMeanzero_fraction_1/Castzero_fraction_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
.dnn/hiddenlayer_1_fraction_of_zero_values/tagsConst*
dtype0*:
value1B/ B)dnn/hiddenlayer_1_fraction_of_zero_values*
_output_shapes
: 
�
)dnn/hiddenlayer_1_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_1_fraction_of_zero_values/tagszero_fraction_1/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_1_activation/tagConst*
dtype0*-
value$B" Bdnn/hiddenlayer_1_activation*
_output_shapes
: 
�
dnn/hiddenlayer_1_activationHistogramSummary dnn/hiddenlayer_1_activation/tag$dnn/hiddenlayer_1/hiddenlayer_1/Relu*
T0*
_output_shapes
: 
�
Adnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB"Q   1   *
_output_shapes
:
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB
 *��[�*
_output_shapes
: 
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB
 *��[>*
_output_shapes
: 
�
Idnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:Q1*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes
: 
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:Q1
�
;dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:Q1
�
 dnn/hiddenlayer_2/weights/part_0
VariableV2*
	container *
_output_shapes

:Q1*
dtype0*
shape
:Q1*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
shared_name 
�
'dnn/hiddenlayer_2/weights/part_0/AssignAssign dnn/hiddenlayer_2/weights/part_0;dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
use_locking(*
T0*
_output_shapes

:Q1
�
%dnn/hiddenlayer_2/weights/part_0/readIdentity dnn/hiddenlayer_2/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:Q1
�
1dnn/hiddenlayer_2/biases/part_0/Initializer/ConstConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
valueB1*    *
_output_shapes
:1
�
dnn/hiddenlayer_2/biases/part_0
VariableV2*
	container *
_output_shapes
:1*
dtype0*
shape:1*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
shared_name 
�
&dnn/hiddenlayer_2/biases/part_0/AssignAssigndnn/hiddenlayer_2/biases/part_01dnn/hiddenlayer_2/biases/part_0/Initializer/Const*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
use_locking(*
T0*
_output_shapes
:1
�
$dnn/hiddenlayer_2/biases/part_0/readIdentitydnn/hiddenlayer_2/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
T0*
_output_shapes
:1
u
dnn/hiddenlayer_2/weightsIdentity%dnn/hiddenlayer_2/weights/part_0/read*
T0*
_output_shapes

:Q1
�
dnn/hiddenlayer_2/MatMulMatMul$dnn/hiddenlayer_1/hiddenlayer_1/Reludnn/hiddenlayer_2/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������1
o
dnn/hiddenlayer_2/biasesIdentity$dnn/hiddenlayer_2/biases/part_0/read*
T0*
_output_shapes
:1
�
dnn/hiddenlayer_2/BiasAddBiasAdddnn/hiddenlayer_2/MatMuldnn/hiddenlayer_2/biases*'
_output_shapes
:���������1*
T0*
data_formatNHWC
y
$dnn/hiddenlayer_2/hiddenlayer_2/ReluReludnn/hiddenlayer_2/BiasAdd*
T0*'
_output_shapes
:���������1
Y
zero_fraction_2/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
zero_fraction_2/EqualEqual$dnn/hiddenlayer_2/hiddenlayer_2/Reluzero_fraction_2/zero*
T0*'
_output_shapes
:���������1
t
zero_fraction_2/CastCastzero_fraction_2/Equal*

DstT0*

SrcT0
*'
_output_shapes
:���������1
f
zero_fraction_2/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
zero_fraction_2/MeanMeanzero_fraction_2/Castzero_fraction_2/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
.dnn/hiddenlayer_2_fraction_of_zero_values/tagsConst*
dtype0*:
value1B/ B)dnn/hiddenlayer_2_fraction_of_zero_values*
_output_shapes
: 
�
)dnn/hiddenlayer_2_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_2_fraction_of_zero_values/tagszero_fraction_2/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_2_activation/tagConst*
dtype0*-
value$B" Bdnn/hiddenlayer_2_activation*
_output_shapes
: 
�
dnn/hiddenlayer_2_activationHistogramSummary dnn/hiddenlayer_2_activation/tag$dnn/hiddenlayer_2/hiddenlayer_2/Relu*
T0*
_output_shapes
: 
�
Adnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
valueB"1      *
_output_shapes
:
�
?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
valueB
 *iʑ�*
_output_shapes
: 
�
?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
valueB
 *iʑ>*
_output_shapes
: 
�
Idnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:1*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0
�
?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
T0*
_output_shapes
: 
�
?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
T0*
_output_shapes

:1
�
;dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
T0*
_output_shapes

:1
�
 dnn/hiddenlayer_3/weights/part_0
VariableV2*
	container *
_output_shapes

:1*
dtype0*
shape
:1*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
shared_name 
�
'dnn/hiddenlayer_3/weights/part_0/AssignAssign dnn/hiddenlayer_3/weights/part_0;dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
use_locking(*
T0*
_output_shapes

:1
�
%dnn/hiddenlayer_3/weights/part_0/readIdentity dnn/hiddenlayer_3/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
T0*
_output_shapes

:1
�
1dnn/hiddenlayer_3/biases/part_0/Initializer/ConstConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
valueB*    *
_output_shapes
:
�
dnn/hiddenlayer_3/biases/part_0
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
shared_name 
�
&dnn/hiddenlayer_3/biases/part_0/AssignAssigndnn/hiddenlayer_3/biases/part_01dnn/hiddenlayer_3/biases/part_0/Initializer/Const*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
use_locking(*
T0*
_output_shapes
:
�
$dnn/hiddenlayer_3/biases/part_0/readIdentitydnn/hiddenlayer_3/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
T0*
_output_shapes
:
u
dnn/hiddenlayer_3/weightsIdentity%dnn/hiddenlayer_3/weights/part_0/read*
T0*
_output_shapes

:1
�
dnn/hiddenlayer_3/MatMulMatMul$dnn/hiddenlayer_2/hiddenlayer_2/Reludnn/hiddenlayer_3/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
o
dnn/hiddenlayer_3/biasesIdentity$dnn/hiddenlayer_3/biases/part_0/read*
T0*
_output_shapes
:
�
dnn/hiddenlayer_3/BiasAddBiasAdddnn/hiddenlayer_3/MatMuldnn/hiddenlayer_3/biases*'
_output_shapes
:���������*
T0*
data_formatNHWC
y
$dnn/hiddenlayer_3/hiddenlayer_3/ReluReludnn/hiddenlayer_3/BiasAdd*
T0*'
_output_shapes
:���������
Y
zero_fraction_3/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
zero_fraction_3/EqualEqual$dnn/hiddenlayer_3/hiddenlayer_3/Reluzero_fraction_3/zero*
T0*'
_output_shapes
:���������
t
zero_fraction_3/CastCastzero_fraction_3/Equal*

DstT0*

SrcT0
*'
_output_shapes
:���������
f
zero_fraction_3/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
zero_fraction_3/MeanMeanzero_fraction_3/Castzero_fraction_3/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
.dnn/hiddenlayer_3_fraction_of_zero_values/tagsConst*
dtype0*:
value1B/ B)dnn/hiddenlayer_3_fraction_of_zero_values*
_output_shapes
: 
�
)dnn/hiddenlayer_3_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_3_fraction_of_zero_values/tagszero_fraction_3/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_3_activation/tagConst*
dtype0*-
value$B" Bdnn/hiddenlayer_3_activation*
_output_shapes
: 
�
dnn/hiddenlayer_3_activationHistogramSummary dnn/hiddenlayer_3_activation/tag$dnn/hiddenlayer_3/hiddenlayer_3/Relu*
T0*
_output_shapes
: 
�
:dnn/logits/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB"      *
_output_shapes
:
�
8dnn/logits/weights/part_0/Initializer/random_uniform/minConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *����*
_output_shapes
: 
�
8dnn/logits/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *���>*
_output_shapes
: 
�
Bdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniform:dnn/logits/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:*
dtype0*
seed2 *

seed *
T0*,
_class"
 loc:@dnn/logits/weights/part_0
�
8dnn/logits/weights/part_0/Initializer/random_uniform/subSub8dnn/logits/weights/part_0/Initializer/random_uniform/max8dnn/logits/weights/part_0/Initializer/random_uniform/min*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes
: 
�
8dnn/logits/weights/part_0/Initializer/random_uniform/mulMulBdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniform8dnn/logits/weights/part_0/Initializer/random_uniform/sub*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

:
�
4dnn/logits/weights/part_0/Initializer/random_uniformAdd8dnn/logits/weights/part_0/Initializer/random_uniform/mul8dnn/logits/weights/part_0/Initializer/random_uniform/min*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

:
�
dnn/logits/weights/part_0
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*,
_class"
 loc:@dnn/logits/weights/part_0*
shared_name 
�
 dnn/logits/weights/part_0/AssignAssigndnn/logits/weights/part_04dnn/logits/weights/part_0/Initializer/random_uniform*
validate_shape(*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking(*
T0*
_output_shapes

:
�
dnn/logits/weights/part_0/readIdentitydnn/logits/weights/part_0*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

:
�
*dnn/logits/biases/part_0/Initializer/ConstConst*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
valueB*    *
_output_shapes
:
�
dnn/logits/biases/part_0
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*+
_class!
loc:@dnn/logits/biases/part_0*
shared_name 
�
dnn/logits/biases/part_0/AssignAssigndnn/logits/biases/part_0*dnn/logits/biases/part_0/Initializer/Const*
validate_shape(*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking(*
T0*
_output_shapes
:
�
dnn/logits/biases/part_0/readIdentitydnn/logits/biases/part_0*+
_class!
loc:@dnn/logits/biases/part_0*
T0*
_output_shapes
:
g
dnn/logits/weightsIdentitydnn/logits/weights/part_0/read*
T0*
_output_shapes

:
�
dnn/logits/MatMulMatMul$dnn/hiddenlayer_3/hiddenlayer_3/Reludnn/logits/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
a
dnn/logits/biasesIdentitydnn/logits/biases/part_0/read*
T0*
_output_shapes
:
�
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/biases*'
_output_shapes
:���������*
T0*
data_formatNHWC
Y
zero_fraction_4/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
z
zero_fraction_4/EqualEqualdnn/logits/BiasAddzero_fraction_4/zero*
T0*'
_output_shapes
:���������
t
zero_fraction_4/CastCastzero_fraction_4/Equal*

DstT0*

SrcT0
*'
_output_shapes
:���������
f
zero_fraction_4/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
zero_fraction_4/MeanMeanzero_fraction_4/Castzero_fraction_4/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
'dnn/logits_fraction_of_zero_values/tagsConst*
dtype0*3
value*B( B"dnn/logits_fraction_of_zero_values*
_output_shapes
: 
�
"dnn/logits_fraction_of_zero_valuesScalarSummary'dnn/logits_fraction_of_zero_values/tagszero_fraction_4/Mean*
T0*
_output_shapes
: 
o
dnn/logits_activation/tagConst*
dtype0*&
valueB Bdnn/logits_activation*
_output_shapes
: 
y
dnn/logits_activationHistogramSummarydnn/logits_activation/tagdnn/logits/BiasAdd*
T0*
_output_shapes
: 
v
predictions/scoresSqueezednn/logits/BiasAdd*
squeeze_dims
*
T0*#
_output_shapes
:���������
x
.training_loss/mean_squared_loss/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
�
*training_loss/mean_squared_loss/ExpandDims
ExpandDimsoutput.training_loss/mean_squared_loss/ExpandDims/dim*

Tdim0*
T0	*'
_output_shapes
:���������
�
'training_loss/mean_squared_loss/ToFloatCast*training_loss/mean_squared_loss/ExpandDims*

DstT0*

SrcT0	*'
_output_shapes
:���������
�
#training_loss/mean_squared_loss/subSubdnn/logits/BiasAdd'training_loss/mean_squared_loss/ToFloat*
T0*'
_output_shapes
:���������
�
training_loss/mean_squared_lossSquare#training_loss/mean_squared_loss/sub*
T0*'
_output_shapes
:���������
d
training_loss/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
training_lossMeantraining_loss/mean_squared_losstraining_loss/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
e
 training_loss/ScalarSummary/tagsConst*
dtype0*
valueB
 Bloss*
_output_shapes
: 
~
training_loss/ScalarSummaryScalarSummary training_loss/ScalarSummary/tagstraining_loss*
T0*
_output_shapes
: 
�
#dnn/learning_rate/Initializer/ConstConst*
dtype0*$
_class
loc:@dnn/learning_rate*
valueB
 *��L=*
_output_shapes
: 
�
dnn/learning_rate
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *$
_class
loc:@dnn/learning_rate*
shared_name 
�
dnn/learning_rate/AssignAssigndnn/learning_rate#dnn/learning_rate/Initializer/Const*
validate_shape(*$
_class
loc:@dnn/learning_rate*
use_locking(*
T0*
_output_shapes
: 
|
dnn/learning_rate/readIdentitydnn/learning_rate*$
_class
loc:@dnn/learning_rate*
T0*
_output_shapes
: 
_
train_op/dnn/gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
a
train_op/dnn/gradients/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
train_op/dnn/gradients/FillFilltrain_op/dnn/gradients/Shapetrain_op/dnn/gradients/Const*
T0*
_output_shapes
: 
�
7train_op/dnn/gradients/training_loss_grad/Reshape/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
�
1train_op/dnn/gradients/training_loss_grad/ReshapeReshapetrain_op/dnn/gradients/Fill7train_op/dnn/gradients/training_loss_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
�
/train_op/dnn/gradients/training_loss_grad/ShapeShapetraining_loss/mean_squared_loss*
out_type0*
T0*
_output_shapes
:
�
.train_op/dnn/gradients/training_loss_grad/TileTile1train_op/dnn/gradients/training_loss_grad/Reshape/train_op/dnn/gradients/training_loss_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
�
1train_op/dnn/gradients/training_loss_grad/Shape_1Shapetraining_loss/mean_squared_loss*
out_type0*
T0*
_output_shapes
:
t
1train_op/dnn/gradients/training_loss_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
y
/train_op/dnn/gradients/training_loss_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
.train_op/dnn/gradients/training_loss_grad/ProdProd1train_op/dnn/gradients/training_loss_grad/Shape_1/train_op/dnn/gradients/training_loss_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
{
1train_op/dnn/gradients/training_loss_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
�
0train_op/dnn/gradients/training_loss_grad/Prod_1Prod1train_op/dnn/gradients/training_loss_grad/Shape_21train_op/dnn/gradients/training_loss_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
u
3train_op/dnn/gradients/training_loss_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
1train_op/dnn/gradients/training_loss_grad/MaximumMaximum0train_op/dnn/gradients/training_loss_grad/Prod_13train_op/dnn/gradients/training_loss_grad/Maximum/y*
T0*
_output_shapes
: 
�
2train_op/dnn/gradients/training_loss_grad/floordivFloorDiv.train_op/dnn/gradients/training_loss_grad/Prod1train_op/dnn/gradients/training_loss_grad/Maximum*
T0*
_output_shapes
: 
�
.train_op/dnn/gradients/training_loss_grad/CastCast2train_op/dnn/gradients/training_loss_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 
�
1train_op/dnn/gradients/training_loss_grad/truedivRealDiv.train_op/dnn/gradients/training_loss_grad/Tile.train_op/dnn/gradients/training_loss_grad/Cast*
T0*'
_output_shapes
:���������
�
Atrain_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul/xConst2^train_op/dnn/gradients/training_loss_grad/truediv*
dtype0*
valueB
 *   @*
_output_shapes
: 
�
?train_op/dnn/gradients/training_loss/mean_squared_loss_grad/mulMulAtrain_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul/x#training_loss/mean_squared_loss/sub*
T0*'
_output_shapes
:���������
�
Atrain_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul_1Mul1train_op/dnn/gradients/training_loss_grad/truediv?train_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul*
T0*'
_output_shapes
:���������
�
Etrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/ShapeShapednn/logits/BiasAdd*
out_type0*
T0*
_output_shapes
:
�
Gtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Shape_1Shape'training_loss/mean_squared_loss/ToFloat*
out_type0*
T0*
_output_shapes
:
�
Utrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgsEtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/ShapeGtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ctrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/SumSumAtrain_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul_1Utrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Gtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/ReshapeReshapeCtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/SumEtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
Etrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Sum_1SumAtrain_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul_1Wtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
Ctrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/NegNegEtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Sum_1*
T0*
_output_shapes
:
�
Itrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape_1ReshapeCtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/NegGtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
Ptrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/group_depsNoOpH^train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/ReshapeJ^train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape_1
�
Xtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/control_dependencyIdentityGtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/ReshapeQ^train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/group_deps*Z
_classP
NLloc:@train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape*
T0*'
_output_shapes
:���������
�
Ztrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/control_dependency_1IdentityItrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape_1Q^train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/group_deps*\
_classR
PNloc:@train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
:train_op/dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGradBiasAddGradXtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/control_dependency*
_output_shapes
:*
T0*
data_formatNHWC
�
?train_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/group_depsNoOpY^train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/control_dependency;^train_op/dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGrad
�
Gtrain_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependencyIdentityXtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/control_dependency@^train_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/group_deps*Z
_classP
NLloc:@train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape*
T0*'
_output_shapes
:���������
�
Itrain_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependency_1Identity:train_op/dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGrad@^train_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/group_deps*M
_classC
A?loc:@train_op/dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
�
4train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMulMatMulGtrain_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependencydnn/logits/weights*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������
�
6train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul_1MatMul$dnn/hiddenlayer_3/hiddenlayer_3/ReluGtrain_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:
�
>train_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/group_depsNoOp5^train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul7^train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul_1
�
Ftrain_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependencyIdentity4train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul?^train_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/group_deps*G
_class=
;9loc:@train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������
�
Htrain_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependency_1Identity6train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul_1?^train_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/group_deps*I
_class?
=;loc:@train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
�
Itrain_op/dnn/gradients/dnn/hiddenlayer_3/hiddenlayer_3/Relu_grad/ReluGradReluGradFtrain_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependency$dnn/hiddenlayer_3/hiddenlayer_3/Relu*
T0*'
_output_shapes
:���������
�
Atrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/BiasAddGradBiasAddGradItrain_op/dnn/gradients/dnn/hiddenlayer_3/hiddenlayer_3/Relu_grad/ReluGrad*
_output_shapes
:*
T0*
data_formatNHWC
�
Ftrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/group_depsNoOpJ^train_op/dnn/gradients/dnn/hiddenlayer_3/hiddenlayer_3/Relu_grad/ReluGradB^train_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/BiasAddGrad
�
Ntrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/control_dependencyIdentityItrain_op/dnn/gradients/dnn/hiddenlayer_3/hiddenlayer_3/Relu_grad/ReluGradG^train_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@train_op/dnn/gradients/dnn/hiddenlayer_3/hiddenlayer_3/Relu_grad/ReluGrad*
T0*'
_output_shapes
:���������
�
Ptrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/control_dependency_1IdentityAtrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/BiasAddGradG^train_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/group_deps*T
_classJ
HFloc:@train_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
�
;train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMulMatMulNtrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/control_dependencydnn/hiddenlayer_3/weights*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������1
�
=train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul_1MatMul$dnn/hiddenlayer_2/hiddenlayer_2/ReluNtrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:1
�
Etrain_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/group_depsNoOp<^train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul>^train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul_1
�
Mtrain_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/control_dependencyIdentity;train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMulF^train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������1
�
Otrain_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/control_dependency_1Identity=train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul_1F^train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

:1
�
Itrain_op/dnn/gradients/dnn/hiddenlayer_2/hiddenlayer_2/Relu_grad/ReluGradReluGradMtrain_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/control_dependency$dnn/hiddenlayer_2/hiddenlayer_2/Relu*
T0*'
_output_shapes
:���������1
�
Atrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/BiasAddGradBiasAddGradItrain_op/dnn/gradients/dnn/hiddenlayer_2/hiddenlayer_2/Relu_grad/ReluGrad*
_output_shapes
:1*
T0*
data_formatNHWC
�
Ftrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/group_depsNoOpJ^train_op/dnn/gradients/dnn/hiddenlayer_2/hiddenlayer_2/Relu_grad/ReluGradB^train_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/BiasAddGrad
�
Ntrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/control_dependencyIdentityItrain_op/dnn/gradients/dnn/hiddenlayer_2/hiddenlayer_2/Relu_grad/ReluGradG^train_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@train_op/dnn/gradients/dnn/hiddenlayer_2/hiddenlayer_2/Relu_grad/ReluGrad*
T0*'
_output_shapes
:���������1
�
Ptrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/control_dependency_1IdentityAtrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/BiasAddGradG^train_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/group_deps*T
_classJ
HFloc:@train_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:1
�
;train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMulMatMulNtrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/control_dependencydnn/hiddenlayer_2/weights*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������Q
�
=train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul_1MatMul$dnn/hiddenlayer_1/hiddenlayer_1/ReluNtrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:Q1
�
Etrain_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/group_depsNoOp<^train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul>^train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul_1
�
Mtrain_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/control_dependencyIdentity;train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMulF^train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������Q
�
Otrain_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/control_dependency_1Identity=train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul_1F^train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:Q1
�
Itrain_op/dnn/gradients/dnn/hiddenlayer_1/hiddenlayer_1/Relu_grad/ReluGradReluGradMtrain_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/control_dependency$dnn/hiddenlayer_1/hiddenlayer_1/Relu*
T0*'
_output_shapes
:���������Q
�
Atrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGradBiasAddGradItrain_op/dnn/gradients/dnn/hiddenlayer_1/hiddenlayer_1/Relu_grad/ReluGrad*
_output_shapes
:Q*
T0*
data_formatNHWC
�
Ftrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/group_depsNoOpJ^train_op/dnn/gradients/dnn/hiddenlayer_1/hiddenlayer_1/Relu_grad/ReluGradB^train_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGrad
�
Ntrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependencyIdentityItrain_op/dnn/gradients/dnn/hiddenlayer_1/hiddenlayer_1/Relu_grad/ReluGradG^train_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@train_op/dnn/gradients/dnn/hiddenlayer_1/hiddenlayer_1/Relu_grad/ReluGrad*
T0*'
_output_shapes
:���������Q
�
Ptrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependency_1IdentityAtrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGradG^train_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/group_deps*T
_classJ
HFloc:@train_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:Q
�
;train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMulMatMulNtrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependencydnn/hiddenlayer_1/weights*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������Q
�
=train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1MatMul$dnn/hiddenlayer_0/hiddenlayer_0/ReluNtrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:QQ
�
Etrain_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/group_depsNoOp<^train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul>^train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1
�
Mtrain_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependencyIdentity;train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMulF^train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������Q
�
Otrain_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependency_1Identity=train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1F^train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:QQ
�
Itrain_op/dnn/gradients/dnn/hiddenlayer_0/hiddenlayer_0/Relu_grad/ReluGradReluGradMtrain_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependency$dnn/hiddenlayer_0/hiddenlayer_0/Relu*
T0*'
_output_shapes
:���������Q
�
Atrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGradBiasAddGradItrain_op/dnn/gradients/dnn/hiddenlayer_0/hiddenlayer_0/Relu_grad/ReluGrad*
_output_shapes
:Q*
T0*
data_formatNHWC
�
Ftrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/group_depsNoOpJ^train_op/dnn/gradients/dnn/hiddenlayer_0/hiddenlayer_0/Relu_grad/ReluGradB^train_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGrad
�
Ntrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependencyIdentityItrain_op/dnn/gradients/dnn/hiddenlayer_0/hiddenlayer_0/Relu_grad/ReluGradG^train_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@train_op/dnn/gradients/dnn/hiddenlayer_0/hiddenlayer_0/Relu_grad/ReluGrad*
T0*'
_output_shapes
:���������Q
�
Ptrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependency_1IdentityAtrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGradG^train_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/group_deps*T
_classJ
HFloc:@train_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:Q
�
;train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMulMatMulNtrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependencydnn/hiddenlayer_0/weights*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������T
�
=train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1MatMul@dnn/input_from_feature_columns/input_from_feature_columns/concatNtrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:TQ
�
Etrain_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/group_depsNoOp<^train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul>^train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1
�
Mtrain_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/control_dependencyIdentity;train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMulF^train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������T
�
Otrain_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/control_dependency_1Identity=train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1F^train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1*
T0*
_output_shapes

:TQ
�
train_op/dnn/ConstConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueBTQ*���=*
_output_shapes

:TQ
�
,dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad
VariableV2*
	container *
_output_shapes

:TQ*
dtype0*
shape
:TQ*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
shared_name 
�
3dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad/AssignAssign,dnn/dnn/hiddenlayer_0/weights/part_0/Adagradtrain_op/dnn/Const*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking(*
T0*
_output_shapes

:TQ
�
1dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad/readIdentity,dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:TQ
�
train_op/dnn/Const_1Const*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
valueBQ*���=*
_output_shapes
:Q
�
+dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad
VariableV2*
	container *
_output_shapes
:Q*
dtype0*
shape:Q*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
shared_name 
�
2dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad/AssignAssign+dnn/dnn/hiddenlayer_0/biases/part_0/Adagradtrain_op/dnn/Const_1*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q
�
0dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad/readIdentity+dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
T0*
_output_shapes
:Q
�
train_op/dnn/Const_2Const*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueBQQ*���=*
_output_shapes

:QQ
�
,dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad
VariableV2*
	container *
_output_shapes

:QQ*
dtype0*
shape
:QQ*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
shared_name 
�
3dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad/AssignAssign,dnn/dnn/hiddenlayer_1/weights/part_0/Adagradtrain_op/dnn/Const_2*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking(*
T0*
_output_shapes

:QQ
�
1dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad/readIdentity,dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:QQ
�
train_op/dnn/Const_3Const*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
valueBQ*���=*
_output_shapes
:Q
�
+dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad
VariableV2*
	container *
_output_shapes
:Q*
dtype0*
shape:Q*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
shared_name 
�
2dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad/AssignAssign+dnn/dnn/hiddenlayer_1/biases/part_0/Adagradtrain_op/dnn/Const_3*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q
�
0dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad/readIdentity+dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
T0*
_output_shapes
:Q
�
train_op/dnn/Const_4Const*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueBQ1*���=*
_output_shapes

:Q1
�
,dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad
VariableV2*
	container *
_output_shapes

:Q1*
dtype0*
shape
:Q1*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
shared_name 
�
3dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad/AssignAssign,dnn/dnn/hiddenlayer_2/weights/part_0/Adagradtrain_op/dnn/Const_4*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
use_locking(*
T0*
_output_shapes

:Q1
�
1dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad/readIdentity,dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:Q1
�
train_op/dnn/Const_5Const*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
valueB1*���=*
_output_shapes
:1
�
+dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad
VariableV2*
	container *
_output_shapes
:1*
dtype0*
shape:1*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
shared_name 
�
2dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad/AssignAssign+dnn/dnn/hiddenlayer_2/biases/part_0/Adagradtrain_op/dnn/Const_5*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
use_locking(*
T0*
_output_shapes
:1
�
0dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad/readIdentity+dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
T0*
_output_shapes
:1
�
train_op/dnn/Const_6Const*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
valueB1*���=*
_output_shapes

:1
�
,dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad
VariableV2*
	container *
_output_shapes

:1*
dtype0*
shape
:1*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
shared_name 
�
3dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad/AssignAssign,dnn/dnn/hiddenlayer_3/weights/part_0/Adagradtrain_op/dnn/Const_6*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
use_locking(*
T0*
_output_shapes

:1
�
1dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad/readIdentity,dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
T0*
_output_shapes

:1
�
train_op/dnn/Const_7Const*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
valueB*���=*
_output_shapes
:
�
+dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
shared_name 
�
2dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad/AssignAssign+dnn/dnn/hiddenlayer_3/biases/part_0/Adagradtrain_op/dnn/Const_7*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
use_locking(*
T0*
_output_shapes
:
�
0dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad/readIdentity+dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
T0*
_output_shapes
:
�
train_op/dnn/Const_8Const*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB*���=*
_output_shapes

:
�
%dnn/dnn/logits/weights/part_0/Adagrad
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*,
_class"
 loc:@dnn/logits/weights/part_0*
shared_name 
�
,dnn/dnn/logits/weights/part_0/Adagrad/AssignAssign%dnn/dnn/logits/weights/part_0/Adagradtrain_op/dnn/Const_8*
validate_shape(*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking(*
T0*
_output_shapes

:
�
*dnn/dnn/logits/weights/part_0/Adagrad/readIdentity%dnn/dnn/logits/weights/part_0/Adagrad*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

:
�
train_op/dnn/Const_9Const*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
valueB*���=*
_output_shapes
:
�
$dnn/dnn/logits/biases/part_0/Adagrad
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*+
_class!
loc:@dnn/logits/biases/part_0*
shared_name 
�
+dnn/dnn/logits/biases/part_0/Adagrad/AssignAssign$dnn/dnn/logits/biases/part_0/Adagradtrain_op/dnn/Const_9*
validate_shape(*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking(*
T0*
_output_shapes
:
�
)dnn/dnn/logits/biases/part_0/Adagrad/readIdentity$dnn/dnn/logits/biases/part_0/Adagrad*+
_class!
loc:@dnn/logits/biases/part_0*
T0*
_output_shapes
:
�
Gtrain_op/dnn/train/update_dnn/hiddenlayer_0/weights/part_0/ApplyAdagradApplyAdagrad dnn/hiddenlayer_0/weights/part_0,dnn/dnn/hiddenlayer_0/weights/part_0/Adagraddnn/learning_rate/readOtrain_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/control_dependency_1*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking( *
T0*
_output_shapes

:TQ
�
Ftrain_op/dnn/train/update_dnn/hiddenlayer_0/biases/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_0/biases/part_0+dnn/dnn/hiddenlayer_0/biases/part_0/Adagraddnn/learning_rate/readPtrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependency_1*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking( *
T0*
_output_shapes
:Q
�
Gtrain_op/dnn/train/update_dnn/hiddenlayer_1/weights/part_0/ApplyAdagradApplyAdagrad dnn/hiddenlayer_1/weights/part_0,dnn/dnn/hiddenlayer_1/weights/part_0/Adagraddnn/learning_rate/readOtrain_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependency_1*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking( *
T0*
_output_shapes

:QQ
�
Ftrain_op/dnn/train/update_dnn/hiddenlayer_1/biases/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_1/biases/part_0+dnn/dnn/hiddenlayer_1/biases/part_0/Adagraddnn/learning_rate/readPtrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependency_1*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking( *
T0*
_output_shapes
:Q
�
Gtrain_op/dnn/train/update_dnn/hiddenlayer_2/weights/part_0/ApplyAdagradApplyAdagrad dnn/hiddenlayer_2/weights/part_0,dnn/dnn/hiddenlayer_2/weights/part_0/Adagraddnn/learning_rate/readOtrain_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/control_dependency_1*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
use_locking( *
T0*
_output_shapes

:Q1
�
Ftrain_op/dnn/train/update_dnn/hiddenlayer_2/biases/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_2/biases/part_0+dnn/dnn/hiddenlayer_2/biases/part_0/Adagraddnn/learning_rate/readPtrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/control_dependency_1*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
use_locking( *
T0*
_output_shapes
:1
�
Gtrain_op/dnn/train/update_dnn/hiddenlayer_3/weights/part_0/ApplyAdagradApplyAdagrad dnn/hiddenlayer_3/weights/part_0,dnn/dnn/hiddenlayer_3/weights/part_0/Adagraddnn/learning_rate/readOtrain_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/control_dependency_1*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
use_locking( *
T0*
_output_shapes

:1
�
Ftrain_op/dnn/train/update_dnn/hiddenlayer_3/biases/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_3/biases/part_0+dnn/dnn/hiddenlayer_3/biases/part_0/Adagraddnn/learning_rate/readPtrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/control_dependency_1*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
use_locking( *
T0*
_output_shapes
:
�
@train_op/dnn/train/update_dnn/logits/weights/part_0/ApplyAdagradApplyAdagraddnn/logits/weights/part_0%dnn/dnn/logits/weights/part_0/Adagraddnn/learning_rate/readHtrain_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependency_1*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking( *
T0*
_output_shapes

:
�
?train_op/dnn/train/update_dnn/logits/biases/part_0/ApplyAdagradApplyAdagraddnn/logits/biases/part_0$dnn/dnn/logits/biases/part_0/Adagraddnn/learning_rate/readItrain_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependency_1*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking( *
T0*
_output_shapes
:
�
train_op/dnn/train/updateNoOpH^train_op/dnn/train/update_dnn/hiddenlayer_0/weights/part_0/ApplyAdagradG^train_op/dnn/train/update_dnn/hiddenlayer_0/biases/part_0/ApplyAdagradH^train_op/dnn/train/update_dnn/hiddenlayer_1/weights/part_0/ApplyAdagradG^train_op/dnn/train/update_dnn/hiddenlayer_1/biases/part_0/ApplyAdagradH^train_op/dnn/train/update_dnn/hiddenlayer_2/weights/part_0/ApplyAdagradG^train_op/dnn/train/update_dnn/hiddenlayer_2/biases/part_0/ApplyAdagradH^train_op/dnn/train/update_dnn/hiddenlayer_3/weights/part_0/ApplyAdagradG^train_op/dnn/train/update_dnn/hiddenlayer_3/biases/part_0/ApplyAdagradA^train_op/dnn/train/update_dnn/logits/weights/part_0/ApplyAdagrad@^train_op/dnn/train/update_dnn/logits/biases/part_0/ApplyAdagrad
�
train_op/dnn/train/valueConst^train_op/dnn/train/update*
dtype0	*
_class
loc:@global_step*
value	B	 R*
_output_shapes
: 
�
train_op/dnn/train	AssignAddglobal_steptrain_op/dnn/train/value*
_class
loc:@global_step*
use_locking( *
T0	*
_output_shapes
: 
�
train_op/dnn/control_dependencyIdentitytraining_loss^train_op/dnn/train* 
_class
loc:@training_loss*
T0*
_output_shapes
: 
r
(metrics/mean_squared_loss/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
�
$metrics/mean_squared_loss/ExpandDims
ExpandDimsoutput(metrics/mean_squared_loss/ExpandDims/dim*

Tdim0*
T0	*'
_output_shapes
:���������
t
*metrics/mean_squared_loss/ExpandDims_1/dimConst*
dtype0*
valueB:*
_output_shapes
:
�
&metrics/mean_squared_loss/ExpandDims_1
ExpandDimspredictions/scores*metrics/mean_squared_loss/ExpandDims_1/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
!metrics/mean_squared_loss/ToFloatCast$metrics/mean_squared_loss/ExpandDims*

DstT0*

SrcT0	*'
_output_shapes
:���������
�
metrics/mean_squared_loss/subSub&metrics/mean_squared_loss/ExpandDims_1!metrics/mean_squared_loss/ToFloat*
T0*'
_output_shapes
:���������
t
metrics/mean_squared_lossSquaremetrics/mean_squared_loss/sub*
T0*'
_output_shapes
:���������
h
metrics/eval_loss/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
metrics/eval_lossMeanmetrics/mean_squared_lossmetrics/eval_loss/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
W
metrics/mean/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/total
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
�
metrics/mean/total/AssignAssignmetrics/mean/totalmetrics/mean/zeros*
validate_shape(*%
_class
loc:@metrics/mean/total*
use_locking(*
T0*
_output_shapes
: 

metrics/mean/total/readIdentitymetrics/mean/total*%
_class
loc:@metrics/mean/total*
T0*
_output_shapes
: 
Y
metrics/mean/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/count
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
�
metrics/mean/count/AssignAssignmetrics/mean/countmetrics/mean/zeros_1*
validate_shape(*%
_class
loc:@metrics/mean/count*
use_locking(*
T0*
_output_shapes
: 

metrics/mean/count/readIdentitymetrics/mean/count*%
_class
loc:@metrics/mean/count*
T0*
_output_shapes
: 
S
metrics/mean/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
a
metrics/mean/ToFloat_1Castmetrics/mean/Size*

DstT0*

SrcT0*
_output_shapes
: 
U
metrics/mean/ConstConst*
dtype0*
valueB *
_output_shapes
: 
|
metrics/mean/SumSummetrics/eval_lossmetrics/mean/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
metrics/mean/AssignAdd	AssignAddmetrics/mean/totalmetrics/mean/Sum*%
_class
loc:@metrics/mean/total*
use_locking( *
T0*
_output_shapes
: 
�
metrics/mean/AssignAdd_1	AssignAddmetrics/mean/countmetrics/mean/ToFloat_1*%
_class
loc:@metrics/mean/count*
use_locking( *
T0*
_output_shapes
: 
[
metrics/mean/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
q
metrics/mean/GreaterGreatermetrics/mean/count/readmetrics/mean/Greater/y*
T0*
_output_shapes
: 
r
metrics/mean/truedivRealDivmetrics/mean/total/readmetrics/mean/count/read*
T0*
_output_shapes
: 
Y
metrics/mean/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 

metrics/mean/valueSelectmetrics/mean/Greatermetrics/mean/truedivmetrics/mean/value/e*
T0*
_output_shapes
: 
]
metrics/mean/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/Greater_1Greatermetrics/mean/AssignAdd_1metrics/mean/Greater_1/y*
T0*
_output_shapes
: 
t
metrics/mean/truediv_1RealDivmetrics/mean/AssignAddmetrics/mean/AssignAdd_1*
T0*
_output_shapes
: 
]
metrics/mean/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
metrics/mean/update_opSelectmetrics/mean/Greater_1metrics/mean/truediv_1metrics/mean/update_op/e*
T0*
_output_shapes
: 
�
initNoOp^global_step/Assign(^dnn/hiddenlayer_0/weights/part_0/Assign'^dnn/hiddenlayer_0/biases/part_0/Assign(^dnn/hiddenlayer_1/weights/part_0/Assign'^dnn/hiddenlayer_1/biases/part_0/Assign(^dnn/hiddenlayer_2/weights/part_0/Assign'^dnn/hiddenlayer_2/biases/part_0/Assign(^dnn/hiddenlayer_3/weights/part_0/Assign'^dnn/hiddenlayer_3/biases/part_0/Assign!^dnn/logits/weights/part_0/Assign ^dnn/logits/biases/part_0/Assign^dnn/learning_rate/Assign4^dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad/Assign3^dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad/Assign4^dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad/Assign3^dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad/Assign4^dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad/Assign3^dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad/Assign4^dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad/Assign3^dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad/Assign-^dnn/dnn/logits/weights/part_0/Adagrad/Assign,^dnn/dnn/logits/biases/part_0/Adagrad/Assign

init_1NoOp
"

group_depsNoOp^init^init_1
�
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_class
loc:@global_step*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitialized dnn/hiddenlayer_0/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializeddnn/hiddenlayer_0/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitialized dnn/hiddenlayer_1/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializeddnn/hiddenlayer_1/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitialized dnn/hiddenlayer_2/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitializeddnn/hiddenlayer_2/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitialized dnn/hiddenlayer_3/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitializeddnn/hiddenlayer_3/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializeddnn/logits/weights/part_0*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializeddnn/logits/biases/part_0*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializeddnn/learning_rate*
dtype0*$
_class
loc:@dnn/learning_rate*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitialized,dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitialized+dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitialized,dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitialized+dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitialized,dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitialized+dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_18IsVariableInitialized,dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_19IsVariableInitialized+dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_20IsVariableInitialized%dnn/dnn/logits/weights/part_0/Adagrad*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_21IsVariableInitialized$dnn/dnn/logits/biases/part_0/Adagrad*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_22IsVariableInitializedmetrics/mean/total*
dtype0*%
_class
loc:@metrics/mean/total*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_23IsVariableInitializedmetrics/mean/count*
dtype0*%
_class
loc:@metrics/mean/count*
_output_shapes
: 
�
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_167report_uninitialized_variables/IsVariableInitialized_177report_uninitialized_variables/IsVariableInitialized_187report_uninitialized_variables/IsVariableInitialized_197report_uninitialized_variables/IsVariableInitialized_207report_uninitialized_variables/IsVariableInitialized_217report_uninitialized_variables/IsVariableInitialized_227report_uninitialized_variables/IsVariableInitialized_23*
N*
T0
*
_output_shapes
:*

axis 
y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:
�
$report_uninitialized_variables/ConstConst*
dtype0*�
value�B�Bglobal_stepB dnn/hiddenlayer_0/weights/part_0Bdnn/hiddenlayer_0/biases/part_0B dnn/hiddenlayer_1/weights/part_0Bdnn/hiddenlayer_1/biases/part_0B dnn/hiddenlayer_2/weights/part_0Bdnn/hiddenlayer_2/biases/part_0B dnn/hiddenlayer_3/weights/part_0Bdnn/hiddenlayer_3/biases/part_0Bdnn/logits/weights/part_0Bdnn/logits/biases/part_0Bdnn/learning_rateB,dnn/dnn/hiddenlayer_0/weights/part_0/AdagradB+dnn/dnn/hiddenlayer_0/biases/part_0/AdagradB,dnn/dnn/hiddenlayer_1/weights/part_0/AdagradB+dnn/dnn/hiddenlayer_1/biases/part_0/AdagradB,dnn/dnn/hiddenlayer_2/weights/part_0/AdagradB+dnn/dnn/hiddenlayer_2/biases/part_0/AdagradB,dnn/dnn/hiddenlayer_3/weights/part_0/AdagradB+dnn/dnn/hiddenlayer_3/biases/part_0/AdagradB%dnn/dnn/logits/weights/part_0/AdagradB$dnn/dnn/logits/biases/part_0/AdagradBmetrics/mean/totalBmetrics/mean/count*
_output_shapes
:
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
�
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
�
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
�
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod*
N*
T0*
_output_shapes
:*

axis 
y
7report_uninitialized_variables/boolean_mask/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
�
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
_output_shapes
:*
T0*
Tshape0
�
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
_output_shapes
:*
T0
*
Tshape0
�
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:���������
�
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:���������
�
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:���������
g
$report_uninitialized_resources/ConstConst*
dtype0*
valueB *
_output_shapes
: 
M
concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*
N*

Tidx0*#
_output_shapes
:���������*
T0
�
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_class
loc:@global_step*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitialized dnn/hiddenlayer_0/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializeddnn/hiddenlayer_0/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitialized dnn/hiddenlayer_1/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitializeddnn/hiddenlayer_1/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitialized dnn/hiddenlayer_2/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitializeddnn/hiddenlayer_2/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitialized dnn/hiddenlayer_3/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitializeddnn/hiddenlayer_3/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_9IsVariableInitializeddnn/logits/weights/part_0*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_10IsVariableInitializeddnn/logits/biases/part_0*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_11IsVariableInitializeddnn/learning_rate*
dtype0*$
_class
loc:@dnn/learning_rate*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_12IsVariableInitialized,dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_13IsVariableInitialized+dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_14IsVariableInitialized,dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_15IsVariableInitialized+dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_16IsVariableInitialized,dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_17IsVariableInitialized+dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_18IsVariableInitialized,dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_19IsVariableInitialized+dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_20IsVariableInitialized%dnn/dnn/logits/weights/part_0/Adagrad*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_21IsVariableInitialized$dnn/dnn/logits/biases/part_0/Adagrad*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
_output_shapes
: 
�

&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_68report_uninitialized_variables_1/IsVariableInitialized_78report_uninitialized_variables_1/IsVariableInitialized_88report_uninitialized_variables_1/IsVariableInitialized_99report_uninitialized_variables_1/IsVariableInitialized_109report_uninitialized_variables_1/IsVariableInitialized_119report_uninitialized_variables_1/IsVariableInitialized_129report_uninitialized_variables_1/IsVariableInitialized_139report_uninitialized_variables_1/IsVariableInitialized_149report_uninitialized_variables_1/IsVariableInitialized_159report_uninitialized_variables_1/IsVariableInitialized_169report_uninitialized_variables_1/IsVariableInitialized_179report_uninitialized_variables_1/IsVariableInitialized_189report_uninitialized_variables_1/IsVariableInitialized_199report_uninitialized_variables_1/IsVariableInitialized_209report_uninitialized_variables_1/IsVariableInitialized_21*
N*
T0
*
_output_shapes
:*

axis 
}
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack*
_output_shapes
:
�
&report_uninitialized_variables_1/ConstConst*
dtype0*�
value�B�Bglobal_stepB dnn/hiddenlayer_0/weights/part_0Bdnn/hiddenlayer_0/biases/part_0B dnn/hiddenlayer_1/weights/part_0Bdnn/hiddenlayer_1/biases/part_0B dnn/hiddenlayer_2/weights/part_0Bdnn/hiddenlayer_2/biases/part_0B dnn/hiddenlayer_3/weights/part_0Bdnn/hiddenlayer_3/biases/part_0Bdnn/logits/weights/part_0Bdnn/logits/biases/part_0Bdnn/learning_rateB,dnn/dnn/hiddenlayer_0/weights/part_0/AdagradB+dnn/dnn/hiddenlayer_0/biases/part_0/AdagradB,dnn/dnn/hiddenlayer_1/weights/part_0/AdagradB+dnn/dnn/hiddenlayer_1/biases/part_0/AdagradB,dnn/dnn/hiddenlayer_2/weights/part_0/AdagradB+dnn/dnn/hiddenlayer_2/biases/part_0/AdagradB,dnn/dnn/hiddenlayer_3/weights/part_0/AdagradB+dnn/dnn/hiddenlayer_3/biases/part_0/AdagradB%dnn/dnn/logits/weights/part_0/AdagradB$dnn/dnn/logits/biases/part_0/Adagrad*
_output_shapes
:
}
3report_uninitialized_variables_1/boolean_mask/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
�
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
�
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
�
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0

5report_uninitialized_variables_1/boolean_mask/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
=report_uninitialized_variables_1/boolean_mask/concat/values_0Pack2report_uninitialized_variables_1/boolean_mask/Prod*
N*
T0*
_output_shapes
:*

axis 
{
9report_uninitialized_variables_1/boolean_mask/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/concat/values_0=report_uninitialized_variables_1/boolean_mask/strided_slice_19report_uninitialized_variables_1/boolean_mask/concat/axis*
N*

Tidx0*
_output_shapes
:*
T0
�
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat*
_output_shapes
:*
T0*
Tshape0
�
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape*
_output_shapes
:*
T0
*
Tshape0
�
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1*'
_output_shapes
:���������
�
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:���������
�
4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:���������
F
init_2NoOp^metrics/mean/total/Assign^metrics/mean/count/Assign

init_all_tablesNoOp
/
group_deps_1NoOp^init_2^init_all_tables
�
Merge/MergeSummaryMergeSummary)dnn/hiddenlayer_0_fraction_of_zero_valuesdnn/hiddenlayer_0_activation)dnn/hiddenlayer_1_fraction_of_zero_valuesdnn/hiddenlayer_1_activation)dnn/hiddenlayer_2_fraction_of_zero_valuesdnn/hiddenlayer_2_activation)dnn/hiddenlayer_3_fraction_of_zero_valuesdnn/hiddenlayer_3_activation"dnn/logits_fraction_of_zero_valuesdnn/logits_activationtraining_loss/ScalarSummary*
_output_shapes
: *
N
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
�
save/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_daedc7951a8147e2967486be9e81b6b0/part*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
Q
save/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*
dtype0*�
value�B�Bdnn/hiddenlayer_0/biasesB$dnn/hiddenlayer_0/biases/t_0/AdagradBdnn/hiddenlayer_0/weightsB%dnn/hiddenlayer_0/weights/t_0/AdagradBdnn/hiddenlayer_1/biasesB$dnn/hiddenlayer_1/biases/t_0/AdagradBdnn/hiddenlayer_1/weightsB%dnn/hiddenlayer_1/weights/t_0/AdagradBdnn/hiddenlayer_2/biasesB$dnn/hiddenlayer_2/biases/t_0/AdagradBdnn/hiddenlayer_2/weightsB%dnn/hiddenlayer_2/weights/t_0/AdagradBdnn/hiddenlayer_3/biasesB$dnn/hiddenlayer_3/biases/t_0/AdagradBdnn/hiddenlayer_3/weightsB%dnn/hiddenlayer_3/weights/t_0/AdagradBdnn/learning_rateBdnn/logits/biasesBdnn/logits/biases/t_0/AdagradBdnn/logits/weightsBdnn/logits/weights/t_0/AdagradBglobal_step*
_output_shapes
:
�
save/SaveV2/shape_and_slicesConst*
dtype0*�
value�B�B81 0,81B81 0,81B84 81 0,84:0,81B84 81 0,84:0,81B81 0,81B81 0,81B81 81 0,81:0,81B81 81 0,81:0,81B49 0,49B49 0,49B81 49 0,81:0,49B81 49 0,81:0,49B25 0,25B25 0,25B49 25 0,49:0,25B49 25 0,49:0,25B B1 0,1B1 0,1B25 1 0,25:0,1B25 1 0,25:0,1B *
_output_shapes
:
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices$dnn/hiddenlayer_0/biases/part_0/read0dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad/read%dnn/hiddenlayer_0/weights/part_0/read1dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad/read$dnn/hiddenlayer_1/biases/part_0/read0dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad/read%dnn/hiddenlayer_1/weights/part_0/read1dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad/read$dnn/hiddenlayer_2/biases/part_0/read0dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad/read%dnn/hiddenlayer_2/weights/part_0/read1dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad/read$dnn/hiddenlayer_3/biases/part_0/read0dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad/read%dnn/hiddenlayer_3/weights/part_0/read1dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad/readdnn/learning_ratednn/logits/biases/part_0/read)dnn/dnn/logits/biases/part_0/Adagrad/readdnn/logits/weights/part_0/read*dnn/dnn/logits/weights/part_0/Adagrad/readglobal_step*$
dtypes
2	
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
T0*
_output_shapes
: 
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
N*
T0*
_output_shapes
:*

axis 
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints*
T0*
_output_shapes
: 
|
save/RestoreV2/tensor_namesConst*
dtype0*-
value$B"Bdnn/hiddenlayer_0/biases*
_output_shapes
:
o
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueBB81 0,81*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/AssignAssigndnn/hiddenlayer_0/biases/part_0save/RestoreV2*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q
�
save/RestoreV2_1/tensor_namesConst*
dtype0*9
value0B.B$dnn/hiddenlayer_0/biases/t_0/Adagrad*
_output_shapes
:
q
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
valueBB81 0,81*
_output_shapes
:
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_1Assign+dnn/dnn/hiddenlayer_0/biases/part_0/Adagradsave/RestoreV2_1*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q

save/RestoreV2_2/tensor_namesConst*
dtype0*.
value%B#Bdnn/hiddenlayer_0/weights*
_output_shapes
:
y
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*$
valueBB84 81 0,84:0,81*
_output_shapes
:
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_2Assign dnn/hiddenlayer_0/weights/part_0save/RestoreV2_2*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking(*
T0*
_output_shapes

:TQ
�
save/RestoreV2_3/tensor_namesConst*
dtype0*:
value1B/B%dnn/hiddenlayer_0/weights/t_0/Adagrad*
_output_shapes
:
y
!save/RestoreV2_3/shape_and_slicesConst*
dtype0*$
valueBB84 81 0,84:0,81*
_output_shapes
:
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_3Assign,dnn/dnn/hiddenlayer_0/weights/part_0/Adagradsave/RestoreV2_3*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking(*
T0*
_output_shapes

:TQ
~
save/RestoreV2_4/tensor_namesConst*
dtype0*-
value$B"Bdnn/hiddenlayer_1/biases*
_output_shapes
:
q
!save/RestoreV2_4/shape_and_slicesConst*
dtype0*
valueBB81 0,81*
_output_shapes
:
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_4Assigndnn/hiddenlayer_1/biases/part_0save/RestoreV2_4*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q
�
save/RestoreV2_5/tensor_namesConst*
dtype0*9
value0B.B$dnn/hiddenlayer_1/biases/t_0/Adagrad*
_output_shapes
:
q
!save/RestoreV2_5/shape_and_slicesConst*
dtype0*
valueBB81 0,81*
_output_shapes
:
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_5Assign+dnn/dnn/hiddenlayer_1/biases/part_0/Adagradsave/RestoreV2_5*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q

save/RestoreV2_6/tensor_namesConst*
dtype0*.
value%B#Bdnn/hiddenlayer_1/weights*
_output_shapes
:
y
!save/RestoreV2_6/shape_and_slicesConst*
dtype0*$
valueBB81 81 0,81:0,81*
_output_shapes
:
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_6Assign dnn/hiddenlayer_1/weights/part_0save/RestoreV2_6*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking(*
T0*
_output_shapes

:QQ
�
save/RestoreV2_7/tensor_namesConst*
dtype0*:
value1B/B%dnn/hiddenlayer_1/weights/t_0/Adagrad*
_output_shapes
:
y
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*$
valueBB81 81 0,81:0,81*
_output_shapes
:
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_7Assign,dnn/dnn/hiddenlayer_1/weights/part_0/Adagradsave/RestoreV2_7*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking(*
T0*
_output_shapes

:QQ
~
save/RestoreV2_8/tensor_namesConst*
dtype0*-
value$B"Bdnn/hiddenlayer_2/biases*
_output_shapes
:
q
!save/RestoreV2_8/shape_and_slicesConst*
dtype0*
valueBB49 0,49*
_output_shapes
:
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_8Assigndnn/hiddenlayer_2/biases/part_0save/RestoreV2_8*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
use_locking(*
T0*
_output_shapes
:1
�
save/RestoreV2_9/tensor_namesConst*
dtype0*9
value0B.B$dnn/hiddenlayer_2/biases/t_0/Adagrad*
_output_shapes
:
q
!save/RestoreV2_9/shape_and_slicesConst*
dtype0*
valueBB49 0,49*
_output_shapes
:
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_9Assign+dnn/dnn/hiddenlayer_2/biases/part_0/Adagradsave/RestoreV2_9*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
use_locking(*
T0*
_output_shapes
:1
�
save/RestoreV2_10/tensor_namesConst*
dtype0*.
value%B#Bdnn/hiddenlayer_2/weights*
_output_shapes
:
z
"save/RestoreV2_10/shape_and_slicesConst*
dtype0*$
valueBB81 49 0,81:0,49*
_output_shapes
:
�
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_10Assign dnn/hiddenlayer_2/weights/part_0save/RestoreV2_10*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
use_locking(*
T0*
_output_shapes

:Q1
�
save/RestoreV2_11/tensor_namesConst*
dtype0*:
value1B/B%dnn/hiddenlayer_2/weights/t_0/Adagrad*
_output_shapes
:
z
"save/RestoreV2_11/shape_and_slicesConst*
dtype0*$
valueBB81 49 0,81:0,49*
_output_shapes
:
�
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_11Assign,dnn/dnn/hiddenlayer_2/weights/part_0/Adagradsave/RestoreV2_11*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
use_locking(*
T0*
_output_shapes

:Q1

save/RestoreV2_12/tensor_namesConst*
dtype0*-
value$B"Bdnn/hiddenlayer_3/biases*
_output_shapes
:
r
"save/RestoreV2_12/shape_and_slicesConst*
dtype0*
valueBB25 0,25*
_output_shapes
:
�
save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_12Assigndnn/hiddenlayer_3/biases/part_0save/RestoreV2_12*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
use_locking(*
T0*
_output_shapes
:
�
save/RestoreV2_13/tensor_namesConst*
dtype0*9
value0B.B$dnn/hiddenlayer_3/biases/t_0/Adagrad*
_output_shapes
:
r
"save/RestoreV2_13/shape_and_slicesConst*
dtype0*
valueBB25 0,25*
_output_shapes
:
�
save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_13Assign+dnn/dnn/hiddenlayer_3/biases/part_0/Adagradsave/RestoreV2_13*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
use_locking(*
T0*
_output_shapes
:
�
save/RestoreV2_14/tensor_namesConst*
dtype0*.
value%B#Bdnn/hiddenlayer_3/weights*
_output_shapes
:
z
"save/RestoreV2_14/shape_and_slicesConst*
dtype0*$
valueBB49 25 0,49:0,25*
_output_shapes
:
�
save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_14Assign dnn/hiddenlayer_3/weights/part_0save/RestoreV2_14*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
use_locking(*
T0*
_output_shapes

:1
�
save/RestoreV2_15/tensor_namesConst*
dtype0*:
value1B/B%dnn/hiddenlayer_3/weights/t_0/Adagrad*
_output_shapes
:
z
"save/RestoreV2_15/shape_and_slicesConst*
dtype0*$
valueBB49 25 0,49:0,25*
_output_shapes
:
�
save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_15Assign,dnn/dnn/hiddenlayer_3/weights/part_0/Adagradsave/RestoreV2_15*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
use_locking(*
T0*
_output_shapes

:1
x
save/RestoreV2_16/tensor_namesConst*
dtype0*&
valueBBdnn/learning_rate*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_16Assigndnn/learning_ratesave/RestoreV2_16*
validate_shape(*$
_class
loc:@dnn/learning_rate*
use_locking(*
T0*
_output_shapes
: 
x
save/RestoreV2_17/tensor_namesConst*
dtype0*&
valueBBdnn/logits/biases*
_output_shapes
:
p
"save/RestoreV2_17/shape_and_slicesConst*
dtype0*
valueBB1 0,1*
_output_shapes
:
�
save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_17Assigndnn/logits/biases/part_0save/RestoreV2_17*
validate_shape(*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking(*
T0*
_output_shapes
:
�
save/RestoreV2_18/tensor_namesConst*
dtype0*2
value)B'Bdnn/logits/biases/t_0/Adagrad*
_output_shapes
:
p
"save/RestoreV2_18/shape_and_slicesConst*
dtype0*
valueBB1 0,1*
_output_shapes
:
�
save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_18Assign$dnn/dnn/logits/biases/part_0/Adagradsave/RestoreV2_18*
validate_shape(*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking(*
T0*
_output_shapes
:
y
save/RestoreV2_19/tensor_namesConst*
dtype0*'
valueBBdnn/logits/weights*
_output_shapes
:
x
"save/RestoreV2_19/shape_and_slicesConst*
dtype0*"
valueBB25 1 0,25:0,1*
_output_shapes
:
�
save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_19Assigndnn/logits/weights/part_0save/RestoreV2_19*
validate_shape(*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking(*
T0*
_output_shapes

:
�
save/RestoreV2_20/tensor_namesConst*
dtype0*3
value*B(Bdnn/logits/weights/t_0/Adagrad*
_output_shapes
:
x
"save/RestoreV2_20/shape_and_slicesConst*
dtype0*"
valueBB25 1 0,25:0,1*
_output_shapes
:
�
save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_20Assign%dnn/dnn/logits/weights/part_0/Adagradsave/RestoreV2_20*
validate_shape(*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking(*
T0*
_output_shapes

:
r
save/RestoreV2_21/tensor_namesConst*
dtype0* 
valueBBglobal_step*
_output_shapes
:
k
"save/RestoreV2_21/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2	*
_output_shapes
:
�
save/Assign_21Assignglobal_stepsave/RestoreV2_21*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21
-
save/restore_allNoOp^save/restore_shard"R�ˢ8�     G'��	#��1�AJ��
�"�"
9
Add
x"T
y"T
z"T"
Ttype:
2	
�
ApplyAdagrad
var"T�
accum"T�
lr"T	
grad"T
out"T�"
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
p
	AssignAdd
ref"T�

value"T

output_ref"T�"
Ttype:
2	"
use_lockingbool( 
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
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
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
A
Equal
x"T
y"T
z
"
Ttype:
2	
�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
�
Gather
params"Tparams
indices"Tindices
output"Tparams"
validate_indicesbool("
Tparamstype"
Tindicestype:
2	
:
Greater
x"T
y"T
z
"
Ttype:
2		
S
HistogramSummary
tag
values"T
summary"
Ttype0:
2		
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtype�
is_initialized
"
dtypetype�


LogicalNot
x

y

o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
b
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
<
Mul
x"T
y"T
z"T"
Ttype:
2	�
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
A
Placeholder
output"dtype"
dtypetype"
shapeshape: 
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
l
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
i
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
?
Select
	condition

t"T
e"T
output"T"	
Ttype
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
0
Square
x"T
y"T"
Ttype:
	2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
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
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
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
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �

Where	
input
	
index	*1.0.12v1.0.0-65-g4763edf-dirty��

global_step/Initializer/ConstConst*
dtype0	*
_class
loc:@global_step*
value	B	 R *
_output_shapes
: 
�
global_step
VariableV2*
	container *
_output_shapes
: *
dtype0	*
shape: *
_class
loc:@global_step*
shared_name 
�
global_step/AssignAssignglobal_stepglobal_step/Initializer/Const*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
T0	*
_output_shapes
: 
W
inputPlaceholder*
dtype0*
shape: *'
_output_shapes
:���������T
T
outputPlaceholder*
dtype0	*
shape: *#
_output_shapes
:���������
�
Kdnn/input_from_feature_columns/input_from_feature_columns/concat/concat_dimConst*
dtype0*
value	B :*
_output_shapes
: 
�
@dnn/input_from_feature_columns/input_from_feature_columns/concatIdentityinput*
T0*'
_output_shapes
:���������T
�
Adnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB"T   Q   *
_output_shapes
:
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *�DC�*
_output_shapes
: 
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueB
 *�DC>*
_output_shapes
: 
�
Idnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:TQ*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes
: 
�
?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:TQ
�
;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:TQ
�
 dnn/hiddenlayer_0/weights/part_0
VariableV2*
	container *
_output_shapes

:TQ*
dtype0*
shape
:TQ*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
shared_name 
�
'dnn/hiddenlayer_0/weights/part_0/AssignAssign dnn/hiddenlayer_0/weights/part_0;dnn/hiddenlayer_0/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking(*
T0*
_output_shapes

:TQ
�
%dnn/hiddenlayer_0/weights/part_0/readIdentity dnn/hiddenlayer_0/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:TQ
�
1dnn/hiddenlayer_0/biases/part_0/Initializer/ConstConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
valueBQ*    *
_output_shapes
:Q
�
dnn/hiddenlayer_0/biases/part_0
VariableV2*
	container *
_output_shapes
:Q*
dtype0*
shape:Q*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
shared_name 
�
&dnn/hiddenlayer_0/biases/part_0/AssignAssigndnn/hiddenlayer_0/biases/part_01dnn/hiddenlayer_0/biases/part_0/Initializer/Const*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q
�
$dnn/hiddenlayer_0/biases/part_0/readIdentitydnn/hiddenlayer_0/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
T0*
_output_shapes
:Q
u
dnn/hiddenlayer_0/weightsIdentity%dnn/hiddenlayer_0/weights/part_0/read*
T0*
_output_shapes

:TQ
�
dnn/hiddenlayer_0/MatMulMatMul@dnn/input_from_feature_columns/input_from_feature_columns/concatdnn/hiddenlayer_0/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������Q
o
dnn/hiddenlayer_0/biasesIdentity$dnn/hiddenlayer_0/biases/part_0/read*
T0*
_output_shapes
:Q
�
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/biases*
data_formatNHWC*
T0*'
_output_shapes
:���������Q
y
$dnn/hiddenlayer_0/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*'
_output_shapes
:���������Q
W
zero_fraction/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
zero_fraction/EqualEqual$dnn/hiddenlayer_0/hiddenlayer_0/Reluzero_fraction/zero*
T0*'
_output_shapes
:���������Q
p
zero_fraction/CastCastzero_fraction/Equal*

DstT0*

SrcT0
*'
_output_shapes
:���������Q
d
zero_fraction/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
zero_fraction/MeanMeanzero_fraction/Castzero_fraction/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
.dnn/hiddenlayer_0_fraction_of_zero_values/tagsConst*
dtype0*:
value1B/ B)dnn/hiddenlayer_0_fraction_of_zero_values*
_output_shapes
: 
�
)dnn/hiddenlayer_0_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_0_fraction_of_zero_values/tagszero_fraction/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_0_activation/tagConst*
dtype0*-
value$B" Bdnn/hiddenlayer_0_activation*
_output_shapes
: 
�
dnn/hiddenlayer_0_activationHistogramSummary dnn/hiddenlayer_0_activation/tag$dnn/hiddenlayer_0/hiddenlayer_0/Relu*
T0*
_output_shapes
: 
�
Adnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB"Q   Q   *
_output_shapes
:
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *�E�*
_output_shapes
: 
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueB
 *�E>*
_output_shapes
: 
�
Idnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:QQ*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes
: 
�
?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:QQ
�
;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:QQ
�
 dnn/hiddenlayer_1/weights/part_0
VariableV2*
	container *
_output_shapes

:QQ*
dtype0*
shape
:QQ*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
shared_name 
�
'dnn/hiddenlayer_1/weights/part_0/AssignAssign dnn/hiddenlayer_1/weights/part_0;dnn/hiddenlayer_1/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking(*
T0*
_output_shapes

:QQ
�
%dnn/hiddenlayer_1/weights/part_0/readIdentity dnn/hiddenlayer_1/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:QQ
�
1dnn/hiddenlayer_1/biases/part_0/Initializer/ConstConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
valueBQ*    *
_output_shapes
:Q
�
dnn/hiddenlayer_1/biases/part_0
VariableV2*
	container *
_output_shapes
:Q*
dtype0*
shape:Q*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
shared_name 
�
&dnn/hiddenlayer_1/biases/part_0/AssignAssigndnn/hiddenlayer_1/biases/part_01dnn/hiddenlayer_1/biases/part_0/Initializer/Const*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q
�
$dnn/hiddenlayer_1/biases/part_0/readIdentitydnn/hiddenlayer_1/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
T0*
_output_shapes
:Q
u
dnn/hiddenlayer_1/weightsIdentity%dnn/hiddenlayer_1/weights/part_0/read*
T0*
_output_shapes

:QQ
�
dnn/hiddenlayer_1/MatMulMatMul$dnn/hiddenlayer_0/hiddenlayer_0/Reludnn/hiddenlayer_1/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������Q
o
dnn/hiddenlayer_1/biasesIdentity$dnn/hiddenlayer_1/biases/part_0/read*
T0*
_output_shapes
:Q
�
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/biases*
data_formatNHWC*
T0*'
_output_shapes
:���������Q
y
$dnn/hiddenlayer_1/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:���������Q
Y
zero_fraction_1/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
zero_fraction_1/EqualEqual$dnn/hiddenlayer_1/hiddenlayer_1/Reluzero_fraction_1/zero*
T0*'
_output_shapes
:���������Q
t
zero_fraction_1/CastCastzero_fraction_1/Equal*

DstT0*

SrcT0
*'
_output_shapes
:���������Q
f
zero_fraction_1/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
zero_fraction_1/MeanMeanzero_fraction_1/Castzero_fraction_1/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
.dnn/hiddenlayer_1_fraction_of_zero_values/tagsConst*
dtype0*:
value1B/ B)dnn/hiddenlayer_1_fraction_of_zero_values*
_output_shapes
: 
�
)dnn/hiddenlayer_1_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_1_fraction_of_zero_values/tagszero_fraction_1/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_1_activation/tagConst*
dtype0*-
value$B" Bdnn/hiddenlayer_1_activation*
_output_shapes
: 
�
dnn/hiddenlayer_1_activationHistogramSummary dnn/hiddenlayer_1_activation/tag$dnn/hiddenlayer_1/hiddenlayer_1/Relu*
T0*
_output_shapes
: 
�
Adnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB"Q   1   *
_output_shapes
:
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB
 *��[�*
_output_shapes
: 
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueB
 *��[>*
_output_shapes
: 
�
Idnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:Q1*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes
: 
�
?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:Q1
�
;dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:Q1
�
 dnn/hiddenlayer_2/weights/part_0
VariableV2*
	container *
_output_shapes

:Q1*
dtype0*
shape
:Q1*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
shared_name 
�
'dnn/hiddenlayer_2/weights/part_0/AssignAssign dnn/hiddenlayer_2/weights/part_0;dnn/hiddenlayer_2/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
use_locking(*
T0*
_output_shapes

:Q1
�
%dnn/hiddenlayer_2/weights/part_0/readIdentity dnn/hiddenlayer_2/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:Q1
�
1dnn/hiddenlayer_2/biases/part_0/Initializer/ConstConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
valueB1*    *
_output_shapes
:1
�
dnn/hiddenlayer_2/biases/part_0
VariableV2*
	container *
_output_shapes
:1*
dtype0*
shape:1*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
shared_name 
�
&dnn/hiddenlayer_2/biases/part_0/AssignAssigndnn/hiddenlayer_2/biases/part_01dnn/hiddenlayer_2/biases/part_0/Initializer/Const*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
use_locking(*
T0*
_output_shapes
:1
�
$dnn/hiddenlayer_2/biases/part_0/readIdentitydnn/hiddenlayer_2/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
T0*
_output_shapes
:1
u
dnn/hiddenlayer_2/weightsIdentity%dnn/hiddenlayer_2/weights/part_0/read*
T0*
_output_shapes

:Q1
�
dnn/hiddenlayer_2/MatMulMatMul$dnn/hiddenlayer_1/hiddenlayer_1/Reludnn/hiddenlayer_2/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������1
o
dnn/hiddenlayer_2/biasesIdentity$dnn/hiddenlayer_2/biases/part_0/read*
T0*
_output_shapes
:1
�
dnn/hiddenlayer_2/BiasAddBiasAdddnn/hiddenlayer_2/MatMuldnn/hiddenlayer_2/biases*
data_formatNHWC*
T0*'
_output_shapes
:���������1
y
$dnn/hiddenlayer_2/hiddenlayer_2/ReluReludnn/hiddenlayer_2/BiasAdd*
T0*'
_output_shapes
:���������1
Y
zero_fraction_2/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
zero_fraction_2/EqualEqual$dnn/hiddenlayer_2/hiddenlayer_2/Reluzero_fraction_2/zero*
T0*'
_output_shapes
:���������1
t
zero_fraction_2/CastCastzero_fraction_2/Equal*

DstT0*

SrcT0
*'
_output_shapes
:���������1
f
zero_fraction_2/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
zero_fraction_2/MeanMeanzero_fraction_2/Castzero_fraction_2/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
.dnn/hiddenlayer_2_fraction_of_zero_values/tagsConst*
dtype0*:
value1B/ B)dnn/hiddenlayer_2_fraction_of_zero_values*
_output_shapes
: 
�
)dnn/hiddenlayer_2_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_2_fraction_of_zero_values/tagszero_fraction_2/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_2_activation/tagConst*
dtype0*-
value$B" Bdnn/hiddenlayer_2_activation*
_output_shapes
: 
�
dnn/hiddenlayer_2_activationHistogramSummary dnn/hiddenlayer_2_activation/tag$dnn/hiddenlayer_2/hiddenlayer_2/Relu*
T0*
_output_shapes
: 
�
Adnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
valueB"1      *
_output_shapes
:
�
?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/minConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
valueB
 *iʑ�*
_output_shapes
: 
�
?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
valueB
 *iʑ>*
_output_shapes
: 
�
Idnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformAdnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:1*
dtype0*
seed2 *

seed *
T0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0
�
?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/subSub?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/max?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
T0*
_output_shapes
: 
�
?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/mulMulIdnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/RandomUniform?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/sub*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
T0*
_output_shapes

:1
�
;dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniformAdd?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/mul?dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform/min*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
T0*
_output_shapes

:1
�
 dnn/hiddenlayer_3/weights/part_0
VariableV2*
	container *
_output_shapes

:1*
dtype0*
shape
:1*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
shared_name 
�
'dnn/hiddenlayer_3/weights/part_0/AssignAssign dnn/hiddenlayer_3/weights/part_0;dnn/hiddenlayer_3/weights/part_0/Initializer/random_uniform*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
use_locking(*
T0*
_output_shapes

:1
�
%dnn/hiddenlayer_3/weights/part_0/readIdentity dnn/hiddenlayer_3/weights/part_0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
T0*
_output_shapes

:1
�
1dnn/hiddenlayer_3/biases/part_0/Initializer/ConstConst*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
valueB*    *
_output_shapes
:
�
dnn/hiddenlayer_3/biases/part_0
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
shared_name 
�
&dnn/hiddenlayer_3/biases/part_0/AssignAssigndnn/hiddenlayer_3/biases/part_01dnn/hiddenlayer_3/biases/part_0/Initializer/Const*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
use_locking(*
T0*
_output_shapes
:
�
$dnn/hiddenlayer_3/biases/part_0/readIdentitydnn/hiddenlayer_3/biases/part_0*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
T0*
_output_shapes
:
u
dnn/hiddenlayer_3/weightsIdentity%dnn/hiddenlayer_3/weights/part_0/read*
T0*
_output_shapes

:1
�
dnn/hiddenlayer_3/MatMulMatMul$dnn/hiddenlayer_2/hiddenlayer_2/Reludnn/hiddenlayer_3/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
o
dnn/hiddenlayer_3/biasesIdentity$dnn/hiddenlayer_3/biases/part_0/read*
T0*
_output_shapes
:
�
dnn/hiddenlayer_3/BiasAddBiasAdddnn/hiddenlayer_3/MatMuldnn/hiddenlayer_3/biases*
data_formatNHWC*
T0*'
_output_shapes
:���������
y
$dnn/hiddenlayer_3/hiddenlayer_3/ReluReludnn/hiddenlayer_3/BiasAdd*
T0*'
_output_shapes
:���������
Y
zero_fraction_3/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
zero_fraction_3/EqualEqual$dnn/hiddenlayer_3/hiddenlayer_3/Reluzero_fraction_3/zero*
T0*'
_output_shapes
:���������
t
zero_fraction_3/CastCastzero_fraction_3/Equal*

DstT0*

SrcT0
*'
_output_shapes
:���������
f
zero_fraction_3/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
zero_fraction_3/MeanMeanzero_fraction_3/Castzero_fraction_3/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
.dnn/hiddenlayer_3_fraction_of_zero_values/tagsConst*
dtype0*:
value1B/ B)dnn/hiddenlayer_3_fraction_of_zero_values*
_output_shapes
: 
�
)dnn/hiddenlayer_3_fraction_of_zero_valuesScalarSummary.dnn/hiddenlayer_3_fraction_of_zero_values/tagszero_fraction_3/Mean*
T0*
_output_shapes
: 
}
 dnn/hiddenlayer_3_activation/tagConst*
dtype0*-
value$B" Bdnn/hiddenlayer_3_activation*
_output_shapes
: 
�
dnn/hiddenlayer_3_activationHistogramSummary dnn/hiddenlayer_3_activation/tag$dnn/hiddenlayer_3/hiddenlayer_3/Relu*
T0*
_output_shapes
: 
�
:dnn/logits/weights/part_0/Initializer/random_uniform/shapeConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB"      *
_output_shapes
:
�
8dnn/logits/weights/part_0/Initializer/random_uniform/minConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *����*
_output_shapes
: 
�
8dnn/logits/weights/part_0/Initializer/random_uniform/maxConst*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB
 *���>*
_output_shapes
: 
�
Bdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniformRandomUniform:dnn/logits/weights/part_0/Initializer/random_uniform/shape*
_output_shapes

:*
dtype0*
seed2 *

seed *
T0*,
_class"
 loc:@dnn/logits/weights/part_0
�
8dnn/logits/weights/part_0/Initializer/random_uniform/subSub8dnn/logits/weights/part_0/Initializer/random_uniform/max8dnn/logits/weights/part_0/Initializer/random_uniform/min*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes
: 
�
8dnn/logits/weights/part_0/Initializer/random_uniform/mulMulBdnn/logits/weights/part_0/Initializer/random_uniform/RandomUniform8dnn/logits/weights/part_0/Initializer/random_uniform/sub*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

:
�
4dnn/logits/weights/part_0/Initializer/random_uniformAdd8dnn/logits/weights/part_0/Initializer/random_uniform/mul8dnn/logits/weights/part_0/Initializer/random_uniform/min*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

:
�
dnn/logits/weights/part_0
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*,
_class"
 loc:@dnn/logits/weights/part_0*
shared_name 
�
 dnn/logits/weights/part_0/AssignAssigndnn/logits/weights/part_04dnn/logits/weights/part_0/Initializer/random_uniform*
validate_shape(*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking(*
T0*
_output_shapes

:
�
dnn/logits/weights/part_0/readIdentitydnn/logits/weights/part_0*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

:
�
*dnn/logits/biases/part_0/Initializer/ConstConst*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
valueB*    *
_output_shapes
:
�
dnn/logits/biases/part_0
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*+
_class!
loc:@dnn/logits/biases/part_0*
shared_name 
�
dnn/logits/biases/part_0/AssignAssigndnn/logits/biases/part_0*dnn/logits/biases/part_0/Initializer/Const*
validate_shape(*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking(*
T0*
_output_shapes
:
�
dnn/logits/biases/part_0/readIdentitydnn/logits/biases/part_0*+
_class!
loc:@dnn/logits/biases/part_0*
T0*
_output_shapes
:
g
dnn/logits/weightsIdentitydnn/logits/weights/part_0/read*
T0*
_output_shapes

:
�
dnn/logits/MatMulMatMul$dnn/hiddenlayer_3/hiddenlayer_3/Reludnn/logits/weights*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
a
dnn/logits/biasesIdentitydnn/logits/biases/part_0/read*
T0*
_output_shapes
:
�
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/biases*
data_formatNHWC*
T0*'
_output_shapes
:���������
Y
zero_fraction_4/zeroConst*
dtype0*
valueB
 *    *
_output_shapes
: 
z
zero_fraction_4/EqualEqualdnn/logits/BiasAddzero_fraction_4/zero*
T0*'
_output_shapes
:���������
t
zero_fraction_4/CastCastzero_fraction_4/Equal*

DstT0*

SrcT0
*'
_output_shapes
:���������
f
zero_fraction_4/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
zero_fraction_4/MeanMeanzero_fraction_4/Castzero_fraction_4/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
'dnn/logits_fraction_of_zero_values/tagsConst*
dtype0*3
value*B( B"dnn/logits_fraction_of_zero_values*
_output_shapes
: 
�
"dnn/logits_fraction_of_zero_valuesScalarSummary'dnn/logits_fraction_of_zero_values/tagszero_fraction_4/Mean*
T0*
_output_shapes
: 
o
dnn/logits_activation/tagConst*
dtype0*&
valueB Bdnn/logits_activation*
_output_shapes
: 
y
dnn/logits_activationHistogramSummarydnn/logits_activation/tagdnn/logits/BiasAdd*
T0*
_output_shapes
: 
v
predictions/scoresSqueezednn/logits/BiasAdd*
squeeze_dims
*
T0*#
_output_shapes
:���������
x
.training_loss/mean_squared_loss/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
�
*training_loss/mean_squared_loss/ExpandDims
ExpandDimsoutput.training_loss/mean_squared_loss/ExpandDims/dim*

Tdim0*
T0	*'
_output_shapes
:���������
�
'training_loss/mean_squared_loss/ToFloatCast*training_loss/mean_squared_loss/ExpandDims*

DstT0*

SrcT0	*'
_output_shapes
:���������
�
#training_loss/mean_squared_loss/subSubdnn/logits/BiasAdd'training_loss/mean_squared_loss/ToFloat*
T0*'
_output_shapes
:���������
�
training_loss/mean_squared_lossSquare#training_loss/mean_squared_loss/sub*
T0*'
_output_shapes
:���������
d
training_loss/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
training_lossMeantraining_loss/mean_squared_losstraining_loss/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
e
 training_loss/ScalarSummary/tagsConst*
dtype0*
valueB
 Bloss*
_output_shapes
: 
~
training_loss/ScalarSummaryScalarSummary training_loss/ScalarSummary/tagstraining_loss*
T0*
_output_shapes
: 
�
#dnn/learning_rate/Initializer/ConstConst*
dtype0*$
_class
loc:@dnn/learning_rate*
valueB
 *��L=*
_output_shapes
: 
�
dnn/learning_rate
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *$
_class
loc:@dnn/learning_rate*
shared_name 
�
dnn/learning_rate/AssignAssigndnn/learning_rate#dnn/learning_rate/Initializer/Const*
validate_shape(*$
_class
loc:@dnn/learning_rate*
use_locking(*
T0*
_output_shapes
: 
|
dnn/learning_rate/readIdentitydnn/learning_rate*$
_class
loc:@dnn/learning_rate*
T0*
_output_shapes
: 
_
train_op/dnn/gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
a
train_op/dnn/gradients/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
train_op/dnn/gradients/FillFilltrain_op/dnn/gradients/Shapetrain_op/dnn/gradients/Const*
T0*
_output_shapes
: 
�
7train_op/dnn/gradients/training_loss_grad/Reshape/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
�
1train_op/dnn/gradients/training_loss_grad/ReshapeReshapetrain_op/dnn/gradients/Fill7train_op/dnn/gradients/training_loss_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes

:
�
/train_op/dnn/gradients/training_loss_grad/ShapeShapetraining_loss/mean_squared_loss*
out_type0*
T0*
_output_shapes
:
�
.train_op/dnn/gradients/training_loss_grad/TileTile1train_op/dnn/gradients/training_loss_grad/Reshape/train_op/dnn/gradients/training_loss_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
�
1train_op/dnn/gradients/training_loss_grad/Shape_1Shapetraining_loss/mean_squared_loss*
out_type0*
T0*
_output_shapes
:
t
1train_op/dnn/gradients/training_loss_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
y
/train_op/dnn/gradients/training_loss_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
.train_op/dnn/gradients/training_loss_grad/ProdProd1train_op/dnn/gradients/training_loss_grad/Shape_1/train_op/dnn/gradients/training_loss_grad/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
{
1train_op/dnn/gradients/training_loss_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
�
0train_op/dnn/gradients/training_loss_grad/Prod_1Prod1train_op/dnn/gradients/training_loss_grad/Shape_21train_op/dnn/gradients/training_loss_grad/Const_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
u
3train_op/dnn/gradients/training_loss_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
1train_op/dnn/gradients/training_loss_grad/MaximumMaximum0train_op/dnn/gradients/training_loss_grad/Prod_13train_op/dnn/gradients/training_loss_grad/Maximum/y*
T0*
_output_shapes
: 
�
2train_op/dnn/gradients/training_loss_grad/floordivFloorDiv.train_op/dnn/gradients/training_loss_grad/Prod1train_op/dnn/gradients/training_loss_grad/Maximum*
T0*
_output_shapes
: 
�
.train_op/dnn/gradients/training_loss_grad/CastCast2train_op/dnn/gradients/training_loss_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 
�
1train_op/dnn/gradients/training_loss_grad/truedivRealDiv.train_op/dnn/gradients/training_loss_grad/Tile.train_op/dnn/gradients/training_loss_grad/Cast*
T0*'
_output_shapes
:���������
�
Atrain_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul/xConst2^train_op/dnn/gradients/training_loss_grad/truediv*
dtype0*
valueB
 *   @*
_output_shapes
: 
�
?train_op/dnn/gradients/training_loss/mean_squared_loss_grad/mulMulAtrain_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul/x#training_loss/mean_squared_loss/sub*
T0*'
_output_shapes
:���������
�
Atrain_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul_1Mul1train_op/dnn/gradients/training_loss_grad/truediv?train_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul*
T0*'
_output_shapes
:���������
�
Etrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/ShapeShapednn/logits/BiasAdd*
out_type0*
T0*
_output_shapes
:
�
Gtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Shape_1Shape'training_loss/mean_squared_loss/ToFloat*
out_type0*
T0*
_output_shapes
:
�
Utrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgsEtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/ShapeGtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ctrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/SumSumAtrain_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul_1Utrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
Gtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/ReshapeReshapeCtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/SumEtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
Etrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Sum_1SumAtrain_op/dnn/gradients/training_loss/mean_squared_loss_grad/mul_1Wtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
Ctrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/NegNegEtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Sum_1*
T0*
_output_shapes
:
�
Itrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape_1ReshapeCtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/NegGtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:���������
�
Ptrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/group_depsNoOpH^train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/ReshapeJ^train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape_1
�
Xtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/control_dependencyIdentityGtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/ReshapeQ^train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/group_deps*Z
_classP
NLloc:@train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape*
T0*'
_output_shapes
:���������
�
Ztrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/control_dependency_1IdentityItrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape_1Q^train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/group_deps*\
_classR
PNloc:@train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
:train_op/dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGradBiasAddGradXtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/control_dependency*
data_formatNHWC*
T0*
_output_shapes
:
�
?train_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/group_depsNoOpY^train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/control_dependency;^train_op/dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGrad
�
Gtrain_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependencyIdentityXtrain_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/tuple/control_dependency@^train_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/group_deps*Z
_classP
NLloc:@train_op/dnn/gradients/training_loss/mean_squared_loss/sub_grad/Reshape*
T0*'
_output_shapes
:���������
�
Itrain_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependency_1Identity:train_op/dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGrad@^train_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/group_deps*M
_classC
A?loc:@train_op/dnn/gradients/dnn/logits/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
�
4train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMulMatMulGtrain_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependencydnn/logits/weights*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������
�
6train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul_1MatMul$dnn/hiddenlayer_3/hiddenlayer_3/ReluGtrain_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:
�
>train_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/group_depsNoOp5^train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul7^train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul_1
�
Ftrain_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependencyIdentity4train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul?^train_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/group_deps*G
_class=
;9loc:@train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������
�
Htrain_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependency_1Identity6train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul_1?^train_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/group_deps*I
_class?
=;loc:@train_op/dnn/gradients/dnn/logits/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
�
Itrain_op/dnn/gradients/dnn/hiddenlayer_3/hiddenlayer_3/Relu_grad/ReluGradReluGradFtrain_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependency$dnn/hiddenlayer_3/hiddenlayer_3/Relu*
T0*'
_output_shapes
:���������
�
Atrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/BiasAddGradBiasAddGradItrain_op/dnn/gradients/dnn/hiddenlayer_3/hiddenlayer_3/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:
�
Ftrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/group_depsNoOpJ^train_op/dnn/gradients/dnn/hiddenlayer_3/hiddenlayer_3/Relu_grad/ReluGradB^train_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/BiasAddGrad
�
Ntrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/control_dependencyIdentityItrain_op/dnn/gradients/dnn/hiddenlayer_3/hiddenlayer_3/Relu_grad/ReluGradG^train_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@train_op/dnn/gradients/dnn/hiddenlayer_3/hiddenlayer_3/Relu_grad/ReluGrad*
T0*'
_output_shapes
:���������
�
Ptrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/control_dependency_1IdentityAtrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/BiasAddGradG^train_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/group_deps*T
_classJ
HFloc:@train_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
�
;train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMulMatMulNtrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/control_dependencydnn/hiddenlayer_3/weights*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������1
�
=train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul_1MatMul$dnn/hiddenlayer_2/hiddenlayer_2/ReluNtrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:1
�
Etrain_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/group_depsNoOp<^train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul>^train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul_1
�
Mtrain_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/control_dependencyIdentity;train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMulF^train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������1
�
Otrain_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/control_dependency_1Identity=train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul_1F^train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@train_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

:1
�
Itrain_op/dnn/gradients/dnn/hiddenlayer_2/hiddenlayer_2/Relu_grad/ReluGradReluGradMtrain_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/control_dependency$dnn/hiddenlayer_2/hiddenlayer_2/Relu*
T0*'
_output_shapes
:���������1
�
Atrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/BiasAddGradBiasAddGradItrain_op/dnn/gradients/dnn/hiddenlayer_2/hiddenlayer_2/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:1
�
Ftrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/group_depsNoOpJ^train_op/dnn/gradients/dnn/hiddenlayer_2/hiddenlayer_2/Relu_grad/ReluGradB^train_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/BiasAddGrad
�
Ntrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/control_dependencyIdentityItrain_op/dnn/gradients/dnn/hiddenlayer_2/hiddenlayer_2/Relu_grad/ReluGradG^train_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@train_op/dnn/gradients/dnn/hiddenlayer_2/hiddenlayer_2/Relu_grad/ReluGrad*
T0*'
_output_shapes
:���������1
�
Ptrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/control_dependency_1IdentityAtrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/BiasAddGradG^train_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/group_deps*T
_classJ
HFloc:@train_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:1
�
;train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMulMatMulNtrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/control_dependencydnn/hiddenlayer_2/weights*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������Q
�
=train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul_1MatMul$dnn/hiddenlayer_1/hiddenlayer_1/ReluNtrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:Q1
�
Etrain_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/group_depsNoOp<^train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul>^train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul_1
�
Mtrain_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/control_dependencyIdentity;train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMulF^train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������Q
�
Otrain_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/control_dependency_1Identity=train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul_1F^train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@train_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:Q1
�
Itrain_op/dnn/gradients/dnn/hiddenlayer_1/hiddenlayer_1/Relu_grad/ReluGradReluGradMtrain_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/control_dependency$dnn/hiddenlayer_1/hiddenlayer_1/Relu*
T0*'
_output_shapes
:���������Q
�
Atrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGradBiasAddGradItrain_op/dnn/gradients/dnn/hiddenlayer_1/hiddenlayer_1/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:Q
�
Ftrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/group_depsNoOpJ^train_op/dnn/gradients/dnn/hiddenlayer_1/hiddenlayer_1/Relu_grad/ReluGradB^train_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGrad
�
Ntrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependencyIdentityItrain_op/dnn/gradients/dnn/hiddenlayer_1/hiddenlayer_1/Relu_grad/ReluGradG^train_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@train_op/dnn/gradients/dnn/hiddenlayer_1/hiddenlayer_1/Relu_grad/ReluGrad*
T0*'
_output_shapes
:���������Q
�
Ptrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependency_1IdentityAtrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGradG^train_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/group_deps*T
_classJ
HFloc:@train_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:Q
�
;train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMulMatMulNtrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependencydnn/hiddenlayer_1/weights*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������Q
�
=train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1MatMul$dnn/hiddenlayer_0/hiddenlayer_0/ReluNtrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:QQ
�
Etrain_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/group_depsNoOp<^train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul>^train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1
�
Mtrain_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependencyIdentity;train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMulF^train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������Q
�
Otrain_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependency_1Identity=train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1F^train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@train_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:QQ
�
Itrain_op/dnn/gradients/dnn/hiddenlayer_0/hiddenlayer_0/Relu_grad/ReluGradReluGradMtrain_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependency$dnn/hiddenlayer_0/hiddenlayer_0/Relu*
T0*'
_output_shapes
:���������Q
�
Atrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGradBiasAddGradItrain_op/dnn/gradients/dnn/hiddenlayer_0/hiddenlayer_0/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:Q
�
Ftrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/group_depsNoOpJ^train_op/dnn/gradients/dnn/hiddenlayer_0/hiddenlayer_0/Relu_grad/ReluGradB^train_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGrad
�
Ntrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependencyIdentityItrain_op/dnn/gradients/dnn/hiddenlayer_0/hiddenlayer_0/Relu_grad/ReluGradG^train_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@train_op/dnn/gradients/dnn/hiddenlayer_0/hiddenlayer_0/Relu_grad/ReluGrad*
T0*'
_output_shapes
:���������Q
�
Ptrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependency_1IdentityAtrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGradG^train_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/group_deps*T
_classJ
HFloc:@train_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:Q
�
;train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMulMatMulNtrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependencydnn/hiddenlayer_0/weights*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:���������T
�
=train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1MatMul@dnn/input_from_feature_columns/input_from_feature_columns/concatNtrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:TQ
�
Etrain_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/group_depsNoOp<^train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul>^train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1
�
Mtrain_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/control_dependencyIdentity;train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMulF^train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/group_deps*N
_classD
B@loc:@train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������T
�
Otrain_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/control_dependency_1Identity=train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1F^train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@train_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/MatMul_1*
T0*
_output_shapes

:TQ
�
train_op/dnn/ConstConst*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
valueBTQ*���=*
_output_shapes

:TQ
�
,dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad
VariableV2*
	container *
_output_shapes

:TQ*
dtype0*
shape
:TQ*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
shared_name 
�
3dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad/AssignAssign,dnn/dnn/hiddenlayer_0/weights/part_0/Adagradtrain_op/dnn/Const*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking(*
T0*
_output_shapes

:TQ
�
1dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad/readIdentity,dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
T0*
_output_shapes

:TQ
�
train_op/dnn/Const_1Const*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
valueBQ*���=*
_output_shapes
:Q
�
+dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad
VariableV2*
	container *
_output_shapes
:Q*
dtype0*
shape:Q*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
shared_name 
�
2dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad/AssignAssign+dnn/dnn/hiddenlayer_0/biases/part_0/Adagradtrain_op/dnn/Const_1*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q
�
0dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad/readIdentity+dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
T0*
_output_shapes
:Q
�
train_op/dnn/Const_2Const*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
valueBQQ*���=*
_output_shapes

:QQ
�
,dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad
VariableV2*
	container *
_output_shapes

:QQ*
dtype0*
shape
:QQ*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
shared_name 
�
3dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad/AssignAssign,dnn/dnn/hiddenlayer_1/weights/part_0/Adagradtrain_op/dnn/Const_2*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking(*
T0*
_output_shapes

:QQ
�
1dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad/readIdentity,dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
T0*
_output_shapes

:QQ
�
train_op/dnn/Const_3Const*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
valueBQ*���=*
_output_shapes
:Q
�
+dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad
VariableV2*
	container *
_output_shapes
:Q*
dtype0*
shape:Q*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
shared_name 
�
2dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad/AssignAssign+dnn/dnn/hiddenlayer_1/biases/part_0/Adagradtrain_op/dnn/Const_3*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q
�
0dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad/readIdentity+dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
T0*
_output_shapes
:Q
�
train_op/dnn/Const_4Const*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
valueBQ1*���=*
_output_shapes

:Q1
�
,dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad
VariableV2*
	container *
_output_shapes

:Q1*
dtype0*
shape
:Q1*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
shared_name 
�
3dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad/AssignAssign,dnn/dnn/hiddenlayer_2/weights/part_0/Adagradtrain_op/dnn/Const_4*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
use_locking(*
T0*
_output_shapes

:Q1
�
1dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad/readIdentity,dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
T0*
_output_shapes

:Q1
�
train_op/dnn/Const_5Const*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
valueB1*���=*
_output_shapes
:1
�
+dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad
VariableV2*
	container *
_output_shapes
:1*
dtype0*
shape:1*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
shared_name 
�
2dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad/AssignAssign+dnn/dnn/hiddenlayer_2/biases/part_0/Adagradtrain_op/dnn/Const_5*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
use_locking(*
T0*
_output_shapes
:1
�
0dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad/readIdentity+dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
T0*
_output_shapes
:1
�
train_op/dnn/Const_6Const*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
valueB1*���=*
_output_shapes

:1
�
,dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad
VariableV2*
	container *
_output_shapes

:1*
dtype0*
shape
:1*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
shared_name 
�
3dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad/AssignAssign,dnn/dnn/hiddenlayer_3/weights/part_0/Adagradtrain_op/dnn/Const_6*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
use_locking(*
T0*
_output_shapes

:1
�
1dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad/readIdentity,dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
T0*
_output_shapes

:1
�
train_op/dnn/Const_7Const*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
valueB*���=*
_output_shapes
:
�
+dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
shared_name 
�
2dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad/AssignAssign+dnn/dnn/hiddenlayer_3/biases/part_0/Adagradtrain_op/dnn/Const_7*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
use_locking(*
T0*
_output_shapes
:
�
0dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad/readIdentity+dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
T0*
_output_shapes
:
�
train_op/dnn/Const_8Const*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
valueB*���=*
_output_shapes

:
�
%dnn/dnn/logits/weights/part_0/Adagrad
VariableV2*
	container *
_output_shapes

:*
dtype0*
shape
:*,
_class"
 loc:@dnn/logits/weights/part_0*
shared_name 
�
,dnn/dnn/logits/weights/part_0/Adagrad/AssignAssign%dnn/dnn/logits/weights/part_0/Adagradtrain_op/dnn/Const_8*
validate_shape(*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking(*
T0*
_output_shapes

:
�
*dnn/dnn/logits/weights/part_0/Adagrad/readIdentity%dnn/dnn/logits/weights/part_0/Adagrad*,
_class"
 loc:@dnn/logits/weights/part_0*
T0*
_output_shapes

:
�
train_op/dnn/Const_9Const*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
valueB*���=*
_output_shapes
:
�
$dnn/dnn/logits/biases/part_0/Adagrad
VariableV2*
	container *
_output_shapes
:*
dtype0*
shape:*+
_class!
loc:@dnn/logits/biases/part_0*
shared_name 
�
+dnn/dnn/logits/biases/part_0/Adagrad/AssignAssign$dnn/dnn/logits/biases/part_0/Adagradtrain_op/dnn/Const_9*
validate_shape(*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking(*
T0*
_output_shapes
:
�
)dnn/dnn/logits/biases/part_0/Adagrad/readIdentity$dnn/dnn/logits/biases/part_0/Adagrad*+
_class!
loc:@dnn/logits/biases/part_0*
T0*
_output_shapes
:
�
Gtrain_op/dnn/train/update_dnn/hiddenlayer_0/weights/part_0/ApplyAdagradApplyAdagrad dnn/hiddenlayer_0/weights/part_0,dnn/dnn/hiddenlayer_0/weights/part_0/Adagraddnn/learning_rate/readOtrain_op/dnn/gradients/dnn/hiddenlayer_0/MatMul_grad/tuple/control_dependency_1*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking( *
T0*
_output_shapes

:TQ
�
Ftrain_op/dnn/train/update_dnn/hiddenlayer_0/biases/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_0/biases/part_0+dnn/dnn/hiddenlayer_0/biases/part_0/Adagraddnn/learning_rate/readPtrain_op/dnn/gradients/dnn/hiddenlayer_0/BiasAdd_grad/tuple/control_dependency_1*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking( *
T0*
_output_shapes
:Q
�
Gtrain_op/dnn/train/update_dnn/hiddenlayer_1/weights/part_0/ApplyAdagradApplyAdagrad dnn/hiddenlayer_1/weights/part_0,dnn/dnn/hiddenlayer_1/weights/part_0/Adagraddnn/learning_rate/readOtrain_op/dnn/gradients/dnn/hiddenlayer_1/MatMul_grad/tuple/control_dependency_1*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking( *
T0*
_output_shapes

:QQ
�
Ftrain_op/dnn/train/update_dnn/hiddenlayer_1/biases/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_1/biases/part_0+dnn/dnn/hiddenlayer_1/biases/part_0/Adagraddnn/learning_rate/readPtrain_op/dnn/gradients/dnn/hiddenlayer_1/BiasAdd_grad/tuple/control_dependency_1*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking( *
T0*
_output_shapes
:Q
�
Gtrain_op/dnn/train/update_dnn/hiddenlayer_2/weights/part_0/ApplyAdagradApplyAdagrad dnn/hiddenlayer_2/weights/part_0,dnn/dnn/hiddenlayer_2/weights/part_0/Adagraddnn/learning_rate/readOtrain_op/dnn/gradients/dnn/hiddenlayer_2/MatMul_grad/tuple/control_dependency_1*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
use_locking( *
T0*
_output_shapes

:Q1
�
Ftrain_op/dnn/train/update_dnn/hiddenlayer_2/biases/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_2/biases/part_0+dnn/dnn/hiddenlayer_2/biases/part_0/Adagraddnn/learning_rate/readPtrain_op/dnn/gradients/dnn/hiddenlayer_2/BiasAdd_grad/tuple/control_dependency_1*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
use_locking( *
T0*
_output_shapes
:1
�
Gtrain_op/dnn/train/update_dnn/hiddenlayer_3/weights/part_0/ApplyAdagradApplyAdagrad dnn/hiddenlayer_3/weights/part_0,dnn/dnn/hiddenlayer_3/weights/part_0/Adagraddnn/learning_rate/readOtrain_op/dnn/gradients/dnn/hiddenlayer_3/MatMul_grad/tuple/control_dependency_1*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
use_locking( *
T0*
_output_shapes

:1
�
Ftrain_op/dnn/train/update_dnn/hiddenlayer_3/biases/part_0/ApplyAdagradApplyAdagraddnn/hiddenlayer_3/biases/part_0+dnn/dnn/hiddenlayer_3/biases/part_0/Adagraddnn/learning_rate/readPtrain_op/dnn/gradients/dnn/hiddenlayer_3/BiasAdd_grad/tuple/control_dependency_1*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
use_locking( *
T0*
_output_shapes
:
�
@train_op/dnn/train/update_dnn/logits/weights/part_0/ApplyAdagradApplyAdagraddnn/logits/weights/part_0%dnn/dnn/logits/weights/part_0/Adagraddnn/learning_rate/readHtrain_op/dnn/gradients/dnn/logits/MatMul_grad/tuple/control_dependency_1*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking( *
T0*
_output_shapes

:
�
?train_op/dnn/train/update_dnn/logits/biases/part_0/ApplyAdagradApplyAdagraddnn/logits/biases/part_0$dnn/dnn/logits/biases/part_0/Adagraddnn/learning_rate/readItrain_op/dnn/gradients/dnn/logits/BiasAdd_grad/tuple/control_dependency_1*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking( *
T0*
_output_shapes
:
�
train_op/dnn/train/updateNoOpH^train_op/dnn/train/update_dnn/hiddenlayer_0/weights/part_0/ApplyAdagradG^train_op/dnn/train/update_dnn/hiddenlayer_0/biases/part_0/ApplyAdagradH^train_op/dnn/train/update_dnn/hiddenlayer_1/weights/part_0/ApplyAdagradG^train_op/dnn/train/update_dnn/hiddenlayer_1/biases/part_0/ApplyAdagradH^train_op/dnn/train/update_dnn/hiddenlayer_2/weights/part_0/ApplyAdagradG^train_op/dnn/train/update_dnn/hiddenlayer_2/biases/part_0/ApplyAdagradH^train_op/dnn/train/update_dnn/hiddenlayer_3/weights/part_0/ApplyAdagradG^train_op/dnn/train/update_dnn/hiddenlayer_3/biases/part_0/ApplyAdagradA^train_op/dnn/train/update_dnn/logits/weights/part_0/ApplyAdagrad@^train_op/dnn/train/update_dnn/logits/biases/part_0/ApplyAdagrad
�
train_op/dnn/train/valueConst^train_op/dnn/train/update*
dtype0	*
_class
loc:@global_step*
value	B	 R*
_output_shapes
: 
�
train_op/dnn/train	AssignAddglobal_steptrain_op/dnn/train/value*
_class
loc:@global_step*
use_locking( *
T0	*
_output_shapes
: 
�
train_op/dnn/control_dependencyIdentitytraining_loss^train_op/dnn/train* 
_class
loc:@training_loss*
T0*
_output_shapes
: 
r
(metrics/mean_squared_loss/ExpandDims/dimConst*
dtype0*
valueB:*
_output_shapes
:
�
$metrics/mean_squared_loss/ExpandDims
ExpandDimsoutput(metrics/mean_squared_loss/ExpandDims/dim*

Tdim0*
T0	*'
_output_shapes
:���������
t
*metrics/mean_squared_loss/ExpandDims_1/dimConst*
dtype0*
valueB:*
_output_shapes
:
�
&metrics/mean_squared_loss/ExpandDims_1
ExpandDimspredictions/scores*metrics/mean_squared_loss/ExpandDims_1/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
!metrics/mean_squared_loss/ToFloatCast$metrics/mean_squared_loss/ExpandDims*

DstT0*

SrcT0	*'
_output_shapes
:���������
�
metrics/mean_squared_loss/subSub&metrics/mean_squared_loss/ExpandDims_1!metrics/mean_squared_loss/ToFloat*
T0*'
_output_shapes
:���������
t
metrics/mean_squared_lossSquaremetrics/mean_squared_loss/sub*
T0*'
_output_shapes
:���������
h
metrics/eval_loss/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
�
metrics/eval_lossMeanmetrics/mean_squared_lossmetrics/eval_loss/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
W
metrics/mean/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/total
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
�
metrics/mean/total/AssignAssignmetrics/mean/totalmetrics/mean/zeros*
validate_shape(*%
_class
loc:@metrics/mean/total*
use_locking(*
T0*
_output_shapes
: 

metrics/mean/total/readIdentitymetrics/mean/total*%
_class
loc:@metrics/mean/total*
T0*
_output_shapes
: 
Y
metrics/mean/zeros_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/count
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
�
metrics/mean/count/AssignAssignmetrics/mean/countmetrics/mean/zeros_1*
validate_shape(*%
_class
loc:@metrics/mean/count*
use_locking(*
T0*
_output_shapes
: 

metrics/mean/count/readIdentitymetrics/mean/count*%
_class
loc:@metrics/mean/count*
T0*
_output_shapes
: 
S
metrics/mean/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
a
metrics/mean/ToFloat_1Castmetrics/mean/Size*

DstT0*

SrcT0*
_output_shapes
: 
U
metrics/mean/ConstConst*
dtype0*
valueB *
_output_shapes
: 
|
metrics/mean/SumSummetrics/eval_lossmetrics/mean/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
metrics/mean/AssignAdd	AssignAddmetrics/mean/totalmetrics/mean/Sum*%
_class
loc:@metrics/mean/total*
use_locking( *
T0*
_output_shapes
: 
�
metrics/mean/AssignAdd_1	AssignAddmetrics/mean/countmetrics/mean/ToFloat_1*%
_class
loc:@metrics/mean/count*
use_locking( *
T0*
_output_shapes
: 
[
metrics/mean/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
q
metrics/mean/GreaterGreatermetrics/mean/count/readmetrics/mean/Greater/y*
T0*
_output_shapes
: 
r
metrics/mean/truedivRealDivmetrics/mean/total/readmetrics/mean/count/read*
T0*
_output_shapes
: 
Y
metrics/mean/value/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 

metrics/mean/valueSelectmetrics/mean/Greatermetrics/mean/truedivmetrics/mean/value/e*
T0*
_output_shapes
: 
]
metrics/mean/Greater_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
v
metrics/mean/Greater_1Greatermetrics/mean/AssignAdd_1metrics/mean/Greater_1/y*
T0*
_output_shapes
: 
t
metrics/mean/truediv_1RealDivmetrics/mean/AssignAddmetrics/mean/AssignAdd_1*
T0*
_output_shapes
: 
]
metrics/mean/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
metrics/mean/update_opSelectmetrics/mean/Greater_1metrics/mean/truediv_1metrics/mean/update_op/e*
T0*
_output_shapes
: 
�
initNoOp^global_step/Assign(^dnn/hiddenlayer_0/weights/part_0/Assign'^dnn/hiddenlayer_0/biases/part_0/Assign(^dnn/hiddenlayer_1/weights/part_0/Assign'^dnn/hiddenlayer_1/biases/part_0/Assign(^dnn/hiddenlayer_2/weights/part_0/Assign'^dnn/hiddenlayer_2/biases/part_0/Assign(^dnn/hiddenlayer_3/weights/part_0/Assign'^dnn/hiddenlayer_3/biases/part_0/Assign!^dnn/logits/weights/part_0/Assign ^dnn/logits/biases/part_0/Assign^dnn/learning_rate/Assign4^dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad/Assign3^dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad/Assign4^dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad/Assign3^dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad/Assign4^dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad/Assign3^dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad/Assign4^dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad/Assign3^dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad/Assign-^dnn/dnn/logits/weights/part_0/Adagrad/Assign,^dnn/dnn/logits/biases/part_0/Adagrad/Assign

init_1NoOp
"

group_depsNoOp^init^init_1
�
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_class
loc:@global_step*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitialized dnn/hiddenlayer_0/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializeddnn/hiddenlayer_0/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitialized dnn/hiddenlayer_1/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializeddnn/hiddenlayer_1/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitialized dnn/hiddenlayer_2/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitializeddnn/hiddenlayer_2/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitialized dnn/hiddenlayer_3/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitializeddnn/hiddenlayer_3/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
_output_shapes
: 
�
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializeddnn/logits/weights/part_0*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializeddnn/logits/biases/part_0*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializeddnn/learning_rate*
dtype0*$
_class
loc:@dnn/learning_rate*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitialized,dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitialized+dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitialized,dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitialized+dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitialized,dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitialized+dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_18IsVariableInitialized,dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_19IsVariableInitialized+dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_20IsVariableInitialized%dnn/dnn/logits/weights/part_0/Adagrad*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_21IsVariableInitialized$dnn/dnn/logits/biases/part_0/Adagrad*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_22IsVariableInitializedmetrics/mean/total*
dtype0*%
_class
loc:@metrics/mean/total*
_output_shapes
: 
�
7report_uninitialized_variables/IsVariableInitialized_23IsVariableInitializedmetrics/mean/count*
dtype0*%
_class
loc:@metrics/mean/count*
_output_shapes
: 
�
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_167report_uninitialized_variables/IsVariableInitialized_177report_uninitialized_variables/IsVariableInitialized_187report_uninitialized_variables/IsVariableInitialized_197report_uninitialized_variables/IsVariableInitialized_207report_uninitialized_variables/IsVariableInitialized_217report_uninitialized_variables/IsVariableInitialized_227report_uninitialized_variables/IsVariableInitialized_23*
_output_shapes
:*

axis *
T0
*
N
y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:
�
$report_uninitialized_variables/ConstConst*
dtype0*�
value�B�Bglobal_stepB dnn/hiddenlayer_0/weights/part_0Bdnn/hiddenlayer_0/biases/part_0B dnn/hiddenlayer_1/weights/part_0Bdnn/hiddenlayer_1/biases/part_0B dnn/hiddenlayer_2/weights/part_0Bdnn/hiddenlayer_2/biases/part_0B dnn/hiddenlayer_3/weights/part_0Bdnn/hiddenlayer_3/biases/part_0Bdnn/logits/weights/part_0Bdnn/logits/biases/part_0Bdnn/learning_rateB,dnn/dnn/hiddenlayer_0/weights/part_0/AdagradB+dnn/dnn/hiddenlayer_0/biases/part_0/AdagradB,dnn/dnn/hiddenlayer_1/weights/part_0/AdagradB+dnn/dnn/hiddenlayer_1/biases/part_0/AdagradB,dnn/dnn/hiddenlayer_2/weights/part_0/AdagradB+dnn/dnn/hiddenlayer_2/biases/part_0/AdagradB,dnn/dnn/hiddenlayer_3/weights/part_0/AdagradB+dnn/dnn/hiddenlayer_3/biases/part_0/AdagradB%dnn/dnn/logits/weights/part_0/AdagradB$dnn/dnn/logits/biases/part_0/AdagradBmetrics/mean/totalBmetrics/mean/count*
_output_shapes
:
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
�
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
�
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
�
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
�
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod*
_output_shapes
:*

axis *
T0*
N
y
7report_uninitialized_variables/boolean_mask/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
�
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
Tshape0*
T0*
_output_shapes
:
�
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
Tshape0*
T0
*
_output_shapes
:
�
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:���������
�
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:���������
�
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:���������
g
$report_uninitialized_resources/ConstConst*
dtype0*
valueB *
_output_shapes
: 
M
concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*#
_output_shapes
:���������*

Tidx0*
T0*
N
�
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_class
loc:@global_step*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitialized dnn/hiddenlayer_0/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializeddnn/hiddenlayer_0/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitialized dnn/hiddenlayer_1/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitializeddnn/hiddenlayer_1/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitialized dnn/hiddenlayer_2/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitializeddnn/hiddenlayer_2/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitialized dnn/hiddenlayer_3/weights/part_0*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitializeddnn/hiddenlayer_3/biases/part_0*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
_output_shapes
: 
�
8report_uninitialized_variables_1/IsVariableInitialized_9IsVariableInitializeddnn/logits/weights/part_0*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_10IsVariableInitializeddnn/logits/biases/part_0*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_11IsVariableInitializeddnn/learning_rate*
dtype0*$
_class
loc:@dnn/learning_rate*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_12IsVariableInitialized,dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_13IsVariableInitialized+dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_14IsVariableInitialized,dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_15IsVariableInitialized+dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_16IsVariableInitialized,dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_17IsVariableInitialized+dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_18IsVariableInitialized,dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad*
dtype0*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_19IsVariableInitialized+dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_20IsVariableInitialized%dnn/dnn/logits/weights/part_0/Adagrad*
dtype0*,
_class"
 loc:@dnn/logits/weights/part_0*
_output_shapes
: 
�
9report_uninitialized_variables_1/IsVariableInitialized_21IsVariableInitialized$dnn/dnn/logits/biases/part_0/Adagrad*
dtype0*+
_class!
loc:@dnn/logits/biases/part_0*
_output_shapes
: 
�

&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_68report_uninitialized_variables_1/IsVariableInitialized_78report_uninitialized_variables_1/IsVariableInitialized_88report_uninitialized_variables_1/IsVariableInitialized_99report_uninitialized_variables_1/IsVariableInitialized_109report_uninitialized_variables_1/IsVariableInitialized_119report_uninitialized_variables_1/IsVariableInitialized_129report_uninitialized_variables_1/IsVariableInitialized_139report_uninitialized_variables_1/IsVariableInitialized_149report_uninitialized_variables_1/IsVariableInitialized_159report_uninitialized_variables_1/IsVariableInitialized_169report_uninitialized_variables_1/IsVariableInitialized_179report_uninitialized_variables_1/IsVariableInitialized_189report_uninitialized_variables_1/IsVariableInitialized_199report_uninitialized_variables_1/IsVariableInitialized_209report_uninitialized_variables_1/IsVariableInitialized_21*
_output_shapes
:*

axis *
T0
*
N
}
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack*
_output_shapes
:
�
&report_uninitialized_variables_1/ConstConst*
dtype0*�
value�B�Bglobal_stepB dnn/hiddenlayer_0/weights/part_0Bdnn/hiddenlayer_0/biases/part_0B dnn/hiddenlayer_1/weights/part_0Bdnn/hiddenlayer_1/biases/part_0B dnn/hiddenlayer_2/weights/part_0Bdnn/hiddenlayer_2/biases/part_0B dnn/hiddenlayer_3/weights/part_0Bdnn/hiddenlayer_3/biases/part_0Bdnn/logits/weights/part_0Bdnn/logits/biases/part_0Bdnn/learning_rateB,dnn/dnn/hiddenlayer_0/weights/part_0/AdagradB+dnn/dnn/hiddenlayer_0/biases/part_0/AdagradB,dnn/dnn/hiddenlayer_1/weights/part_0/AdagradB+dnn/dnn/hiddenlayer_1/biases/part_0/AdagradB,dnn/dnn/hiddenlayer_2/weights/part_0/AdagradB+dnn/dnn/hiddenlayer_2/biases/part_0/AdagradB,dnn/dnn/hiddenlayer_3/weights/part_0/AdagradB+dnn/dnn/hiddenlayer_3/biases/part_0/AdagradB%dnn/dnn/logits/weights/part_0/AdagradB$dnn/dnn/logits/biases/part_0/Adagrad*
_output_shapes
:
}
3report_uninitialized_variables_1/boolean_mask/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
�
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
�
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst*
dtype0*
valueB: *
_output_shapes
:
�
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 

5report_uninitialized_variables_1/boolean_mask/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
�
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst*
dtype0*
valueB:*
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
�
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
=report_uninitialized_variables_1/boolean_mask/concat/values_0Pack2report_uninitialized_variables_1/boolean_mask/Prod*
_output_shapes
:*

axis *
T0*
N
{
9report_uninitialized_variables_1/boolean_mask/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/concat/values_0=report_uninitialized_variables_1/boolean_mask/strided_slice_19report_uninitialized_variables_1/boolean_mask/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
�
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat*
Tshape0*
T0*
_output_shapes
:
�
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst*
dtype0*
valueB:
���������*
_output_shapes
:
�
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape*
Tshape0*
T0
*
_output_shapes
:
�
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1*'
_output_shapes
:���������
�
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where*
squeeze_dims
*
T0	*#
_output_shapes
:���������
�
4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze*
validate_indices(*
Tparams0*
Tindices0	*#
_output_shapes
:���������
F
init_2NoOp^metrics/mean/total/Assign^metrics/mean/count/Assign

init_all_tablesNoOp
/
group_deps_1NoOp^init_2^init_all_tables
�
Merge/MergeSummaryMergeSummary)dnn/hiddenlayer_0_fraction_of_zero_valuesdnn/hiddenlayer_0_activation)dnn/hiddenlayer_1_fraction_of_zero_valuesdnn/hiddenlayer_1_activation)dnn/hiddenlayer_2_fraction_of_zero_valuesdnn/hiddenlayer_2_activation)dnn/hiddenlayer_3_fraction_of_zero_valuesdnn/hiddenlayer_3_activation"dnn/logits_fraction_of_zero_valuesdnn/logits_activationtraining_loss/ScalarSummary*
N*
_output_shapes
: 
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
�
save/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_daedc7951a8147e2967486be9e81b6b0/part*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
Q
save/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*
dtype0*�
value�B�Bdnn/hiddenlayer_0/biasesB$dnn/hiddenlayer_0/biases/t_0/AdagradBdnn/hiddenlayer_0/weightsB%dnn/hiddenlayer_0/weights/t_0/AdagradBdnn/hiddenlayer_1/biasesB$dnn/hiddenlayer_1/biases/t_0/AdagradBdnn/hiddenlayer_1/weightsB%dnn/hiddenlayer_1/weights/t_0/AdagradBdnn/hiddenlayer_2/biasesB$dnn/hiddenlayer_2/biases/t_0/AdagradBdnn/hiddenlayer_2/weightsB%dnn/hiddenlayer_2/weights/t_0/AdagradBdnn/hiddenlayer_3/biasesB$dnn/hiddenlayer_3/biases/t_0/AdagradBdnn/hiddenlayer_3/weightsB%dnn/hiddenlayer_3/weights/t_0/AdagradBdnn/learning_rateBdnn/logits/biasesBdnn/logits/biases/t_0/AdagradBdnn/logits/weightsBdnn/logits/weights/t_0/AdagradBglobal_step*
_output_shapes
:
�
save/SaveV2/shape_and_slicesConst*
dtype0*�
value�B�B81 0,81B81 0,81B84 81 0,84:0,81B84 81 0,84:0,81B81 0,81B81 0,81B81 81 0,81:0,81B81 81 0,81:0,81B49 0,49B49 0,49B81 49 0,81:0,49B81 49 0,81:0,49B25 0,25B25 0,25B49 25 0,49:0,25B49 25 0,49:0,25B B1 0,1B1 0,1B25 1 0,25:0,1B25 1 0,25:0,1B *
_output_shapes
:
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices$dnn/hiddenlayer_0/biases/part_0/read0dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad/read%dnn/hiddenlayer_0/weights/part_0/read1dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad/read$dnn/hiddenlayer_1/biases/part_0/read0dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad/read%dnn/hiddenlayer_1/weights/part_0/read1dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad/read$dnn/hiddenlayer_2/biases/part_0/read0dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad/read%dnn/hiddenlayer_2/weights/part_0/read1dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad/read$dnn/hiddenlayer_3/biases/part_0/read0dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad/read%dnn/hiddenlayer_3/weights/part_0/read1dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad/readdnn/learning_ratednn/logits/biases/part_0/read)dnn/dnn/logits/biases/part_0/Adagrad/readdnn/logits/weights/part_0/read*dnn/dnn/logits/weights/part_0/Adagrad/readglobal_step*$
dtypes
2	
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
T0*
_output_shapes
: 
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
_output_shapes
:*

axis *
T0*
N
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints*
T0*
_output_shapes
: 
|
save/RestoreV2/tensor_namesConst*
dtype0*-
value$B"Bdnn/hiddenlayer_0/biases*
_output_shapes
:
o
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueBB81 0,81*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/AssignAssigndnn/hiddenlayer_0/biases/part_0save/RestoreV2*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q
�
save/RestoreV2_1/tensor_namesConst*
dtype0*9
value0B.B$dnn/hiddenlayer_0/biases/t_0/Adagrad*
_output_shapes
:
q
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
valueBB81 0,81*
_output_shapes
:
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_1Assign+dnn/dnn/hiddenlayer_0/biases/part_0/Adagradsave/RestoreV2_1*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_0/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q

save/RestoreV2_2/tensor_namesConst*
dtype0*.
value%B#Bdnn/hiddenlayer_0/weights*
_output_shapes
:
y
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*$
valueBB84 81 0,84:0,81*
_output_shapes
:
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_2Assign dnn/hiddenlayer_0/weights/part_0save/RestoreV2_2*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking(*
T0*
_output_shapes

:TQ
�
save/RestoreV2_3/tensor_namesConst*
dtype0*:
value1B/B%dnn/hiddenlayer_0/weights/t_0/Adagrad*
_output_shapes
:
y
!save/RestoreV2_3/shape_and_slicesConst*
dtype0*$
valueBB84 81 0,84:0,81*
_output_shapes
:
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_3Assign,dnn/dnn/hiddenlayer_0/weights/part_0/Adagradsave/RestoreV2_3*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_0/weights/part_0*
use_locking(*
T0*
_output_shapes

:TQ
~
save/RestoreV2_4/tensor_namesConst*
dtype0*-
value$B"Bdnn/hiddenlayer_1/biases*
_output_shapes
:
q
!save/RestoreV2_4/shape_and_slicesConst*
dtype0*
valueBB81 0,81*
_output_shapes
:
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_4Assigndnn/hiddenlayer_1/biases/part_0save/RestoreV2_4*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q
�
save/RestoreV2_5/tensor_namesConst*
dtype0*9
value0B.B$dnn/hiddenlayer_1/biases/t_0/Adagrad*
_output_shapes
:
q
!save/RestoreV2_5/shape_and_slicesConst*
dtype0*
valueBB81 0,81*
_output_shapes
:
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_5Assign+dnn/dnn/hiddenlayer_1/biases/part_0/Adagradsave/RestoreV2_5*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_1/biases/part_0*
use_locking(*
T0*
_output_shapes
:Q

save/RestoreV2_6/tensor_namesConst*
dtype0*.
value%B#Bdnn/hiddenlayer_1/weights*
_output_shapes
:
y
!save/RestoreV2_6/shape_and_slicesConst*
dtype0*$
valueBB81 81 0,81:0,81*
_output_shapes
:
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_6Assign dnn/hiddenlayer_1/weights/part_0save/RestoreV2_6*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking(*
T0*
_output_shapes

:QQ
�
save/RestoreV2_7/tensor_namesConst*
dtype0*:
value1B/B%dnn/hiddenlayer_1/weights/t_0/Adagrad*
_output_shapes
:
y
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*$
valueBB81 81 0,81:0,81*
_output_shapes
:
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_7Assign,dnn/dnn/hiddenlayer_1/weights/part_0/Adagradsave/RestoreV2_7*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_1/weights/part_0*
use_locking(*
T0*
_output_shapes

:QQ
~
save/RestoreV2_8/tensor_namesConst*
dtype0*-
value$B"Bdnn/hiddenlayer_2/biases*
_output_shapes
:
q
!save/RestoreV2_8/shape_and_slicesConst*
dtype0*
valueBB49 0,49*
_output_shapes
:
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_8Assigndnn/hiddenlayer_2/biases/part_0save/RestoreV2_8*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
use_locking(*
T0*
_output_shapes
:1
�
save/RestoreV2_9/tensor_namesConst*
dtype0*9
value0B.B$dnn/hiddenlayer_2/biases/t_0/Adagrad*
_output_shapes
:
q
!save/RestoreV2_9/shape_and_slicesConst*
dtype0*
valueBB49 0,49*
_output_shapes
:
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_9Assign+dnn/dnn/hiddenlayer_2/biases/part_0/Adagradsave/RestoreV2_9*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_2/biases/part_0*
use_locking(*
T0*
_output_shapes
:1
�
save/RestoreV2_10/tensor_namesConst*
dtype0*.
value%B#Bdnn/hiddenlayer_2/weights*
_output_shapes
:
z
"save/RestoreV2_10/shape_and_slicesConst*
dtype0*$
valueBB81 49 0,81:0,49*
_output_shapes
:
�
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_10Assign dnn/hiddenlayer_2/weights/part_0save/RestoreV2_10*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
use_locking(*
T0*
_output_shapes

:Q1
�
save/RestoreV2_11/tensor_namesConst*
dtype0*:
value1B/B%dnn/hiddenlayer_2/weights/t_0/Adagrad*
_output_shapes
:
z
"save/RestoreV2_11/shape_and_slicesConst*
dtype0*$
valueBB81 49 0,81:0,49*
_output_shapes
:
�
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_11Assign,dnn/dnn/hiddenlayer_2/weights/part_0/Adagradsave/RestoreV2_11*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_2/weights/part_0*
use_locking(*
T0*
_output_shapes

:Q1

save/RestoreV2_12/tensor_namesConst*
dtype0*-
value$B"Bdnn/hiddenlayer_3/biases*
_output_shapes
:
r
"save/RestoreV2_12/shape_and_slicesConst*
dtype0*
valueBB25 0,25*
_output_shapes
:
�
save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_12Assigndnn/hiddenlayer_3/biases/part_0save/RestoreV2_12*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
use_locking(*
T0*
_output_shapes
:
�
save/RestoreV2_13/tensor_namesConst*
dtype0*9
value0B.B$dnn/hiddenlayer_3/biases/t_0/Adagrad*
_output_shapes
:
r
"save/RestoreV2_13/shape_and_slicesConst*
dtype0*
valueBB25 0,25*
_output_shapes
:
�
save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_13Assign+dnn/dnn/hiddenlayer_3/biases/part_0/Adagradsave/RestoreV2_13*
validate_shape(*2
_class(
&$loc:@dnn/hiddenlayer_3/biases/part_0*
use_locking(*
T0*
_output_shapes
:
�
save/RestoreV2_14/tensor_namesConst*
dtype0*.
value%B#Bdnn/hiddenlayer_3/weights*
_output_shapes
:
z
"save/RestoreV2_14/shape_and_slicesConst*
dtype0*$
valueBB49 25 0,49:0,25*
_output_shapes
:
�
save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_14Assign dnn/hiddenlayer_3/weights/part_0save/RestoreV2_14*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
use_locking(*
T0*
_output_shapes

:1
�
save/RestoreV2_15/tensor_namesConst*
dtype0*:
value1B/B%dnn/hiddenlayer_3/weights/t_0/Adagrad*
_output_shapes
:
z
"save/RestoreV2_15/shape_and_slicesConst*
dtype0*$
valueBB49 25 0,49:0,25*
_output_shapes
:
�
save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_15Assign,dnn/dnn/hiddenlayer_3/weights/part_0/Adagradsave/RestoreV2_15*
validate_shape(*3
_class)
'%loc:@dnn/hiddenlayer_3/weights/part_0*
use_locking(*
T0*
_output_shapes

:1
x
save/RestoreV2_16/tensor_namesConst*
dtype0*&
valueBBdnn/learning_rate*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_16Assigndnn/learning_ratesave/RestoreV2_16*
validate_shape(*$
_class
loc:@dnn/learning_rate*
use_locking(*
T0*
_output_shapes
: 
x
save/RestoreV2_17/tensor_namesConst*
dtype0*&
valueBBdnn/logits/biases*
_output_shapes
:
p
"save/RestoreV2_17/shape_and_slicesConst*
dtype0*
valueBB1 0,1*
_output_shapes
:
�
save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_17Assigndnn/logits/biases/part_0save/RestoreV2_17*
validate_shape(*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking(*
T0*
_output_shapes
:
�
save/RestoreV2_18/tensor_namesConst*
dtype0*2
value)B'Bdnn/logits/biases/t_0/Adagrad*
_output_shapes
:
p
"save/RestoreV2_18/shape_and_slicesConst*
dtype0*
valueBB1 0,1*
_output_shapes
:
�
save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_18Assign$dnn/dnn/logits/biases/part_0/Adagradsave/RestoreV2_18*
validate_shape(*+
_class!
loc:@dnn/logits/biases/part_0*
use_locking(*
T0*
_output_shapes
:
y
save/RestoreV2_19/tensor_namesConst*
dtype0*'
valueBBdnn/logits/weights*
_output_shapes
:
x
"save/RestoreV2_19/shape_and_slicesConst*
dtype0*"
valueBB25 1 0,25:0,1*
_output_shapes
:
�
save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_19Assigndnn/logits/weights/part_0save/RestoreV2_19*
validate_shape(*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking(*
T0*
_output_shapes

:
�
save/RestoreV2_20/tensor_namesConst*
dtype0*3
value*B(Bdnn/logits/weights/t_0/Adagrad*
_output_shapes
:
x
"save/RestoreV2_20/shape_and_slicesConst*
dtype0*"
valueBB25 1 0,25:0,1*
_output_shapes
:
�
save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_20Assign%dnn/dnn/logits/weights/part_0/Adagradsave/RestoreV2_20*
validate_shape(*,
_class"
 loc:@dnn/logits/weights/part_0*
use_locking(*
T0*
_output_shapes

:
r
save/RestoreV2_21/tensor_namesConst*
dtype0* 
valueBBglobal_step*
_output_shapes
:
k
"save/RestoreV2_21/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
�
save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2	*
_output_shapes
:
�
save/Assign_21Assignglobal_stepsave/RestoreV2_21*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"
losses

training_loss:0" 
global_step

global_step:0"�
trainable_variables��
�
"dnn/hiddenlayer_0/weights/part_0:0'dnn/hiddenlayer_0/weights/part_0/Assign'dnn/hiddenlayer_0/weights/part_0/read:0"'
dnn/hiddenlayer_0/weightsTQ  "TQ
�
!dnn/hiddenlayer_0/biases/part_0:0&dnn/hiddenlayer_0/biases/part_0/Assign&dnn/hiddenlayer_0/biases/part_0/read:0"#
dnn/hiddenlayer_0/biasesQ "Q
�
"dnn/hiddenlayer_1/weights/part_0:0'dnn/hiddenlayer_1/weights/part_0/Assign'dnn/hiddenlayer_1/weights/part_0/read:0"'
dnn/hiddenlayer_1/weightsQQ  "QQ
�
!dnn/hiddenlayer_1/biases/part_0:0&dnn/hiddenlayer_1/biases/part_0/Assign&dnn/hiddenlayer_1/biases/part_0/read:0"#
dnn/hiddenlayer_1/biasesQ "Q
�
"dnn/hiddenlayer_2/weights/part_0:0'dnn/hiddenlayer_2/weights/part_0/Assign'dnn/hiddenlayer_2/weights/part_0/read:0"'
dnn/hiddenlayer_2/weightsQ1  "Q1
�
!dnn/hiddenlayer_2/biases/part_0:0&dnn/hiddenlayer_2/biases/part_0/Assign&dnn/hiddenlayer_2/biases/part_0/read:0"#
dnn/hiddenlayer_2/biases1 "1
�
"dnn/hiddenlayer_3/weights/part_0:0'dnn/hiddenlayer_3/weights/part_0/Assign'dnn/hiddenlayer_3/weights/part_0/read:0"'
dnn/hiddenlayer_3/weights1  "1
�
!dnn/hiddenlayer_3/biases/part_0:0&dnn/hiddenlayer_3/biases/part_0/Assign&dnn/hiddenlayer_3/biases/part_0/read:0"#
dnn/hiddenlayer_3/biases "
�
dnn/logits/weights/part_0:0 dnn/logits/weights/part_0/Assign dnn/logits/weights/part_0/read:0" 
dnn/logits/weights  "
|
dnn/logits/biases/part_0:0dnn/logits/biases/part_0/Assigndnn/logits/biases/part_0/read:0"
dnn/logits/biases ""!
local_init_op

group_deps_1"�
	variables��
7
global_step:0global_step/Assignglobal_step/read:0
�
"dnn/hiddenlayer_0/weights/part_0:0'dnn/hiddenlayer_0/weights/part_0/Assign'dnn/hiddenlayer_0/weights/part_0/read:0"'
dnn/hiddenlayer_0/weightsTQ  "TQ
�
!dnn/hiddenlayer_0/biases/part_0:0&dnn/hiddenlayer_0/biases/part_0/Assign&dnn/hiddenlayer_0/biases/part_0/read:0"#
dnn/hiddenlayer_0/biasesQ "Q
�
"dnn/hiddenlayer_1/weights/part_0:0'dnn/hiddenlayer_1/weights/part_0/Assign'dnn/hiddenlayer_1/weights/part_0/read:0"'
dnn/hiddenlayer_1/weightsQQ  "QQ
�
!dnn/hiddenlayer_1/biases/part_0:0&dnn/hiddenlayer_1/biases/part_0/Assign&dnn/hiddenlayer_1/biases/part_0/read:0"#
dnn/hiddenlayer_1/biasesQ "Q
�
"dnn/hiddenlayer_2/weights/part_0:0'dnn/hiddenlayer_2/weights/part_0/Assign'dnn/hiddenlayer_2/weights/part_0/read:0"'
dnn/hiddenlayer_2/weightsQ1  "Q1
�
!dnn/hiddenlayer_2/biases/part_0:0&dnn/hiddenlayer_2/biases/part_0/Assign&dnn/hiddenlayer_2/biases/part_0/read:0"#
dnn/hiddenlayer_2/biases1 "1
�
"dnn/hiddenlayer_3/weights/part_0:0'dnn/hiddenlayer_3/weights/part_0/Assign'dnn/hiddenlayer_3/weights/part_0/read:0"'
dnn/hiddenlayer_3/weights1  "1
�
!dnn/hiddenlayer_3/biases/part_0:0&dnn/hiddenlayer_3/biases/part_0/Assign&dnn/hiddenlayer_3/biases/part_0/read:0"#
dnn/hiddenlayer_3/biases "
�
dnn/logits/weights/part_0:0 dnn/logits/weights/part_0/Assign dnn/logits/weights/part_0/read:0" 
dnn/logits/weights  "
|
dnn/logits/biases/part_0:0dnn/logits/biases/part_0/Assigndnn/logits/biases/part_0/read:0"
dnn/logits/biases "
I
dnn/learning_rate:0dnn/learning_rate/Assigndnn/learning_rate/read:0
�
.dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad:03dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad/Assign3dnn/dnn/hiddenlayer_0/weights/part_0/Adagrad/read:0"3
%dnn/hiddenlayer_0/weights/t_0/AdagradTQ  "TQ
�
-dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad:02dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad/Assign2dnn/dnn/hiddenlayer_0/biases/part_0/Adagrad/read:0"/
$dnn/hiddenlayer_0/biases/t_0/AdagradQ "Q
�
.dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad:03dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad/Assign3dnn/dnn/hiddenlayer_1/weights/part_0/Adagrad/read:0"3
%dnn/hiddenlayer_1/weights/t_0/AdagradQQ  "QQ
�
-dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad:02dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad/Assign2dnn/dnn/hiddenlayer_1/biases/part_0/Adagrad/read:0"/
$dnn/hiddenlayer_1/biases/t_0/AdagradQ "Q
�
.dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad:03dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad/Assign3dnn/dnn/hiddenlayer_2/weights/part_0/Adagrad/read:0"3
%dnn/hiddenlayer_2/weights/t_0/AdagradQ1  "Q1
�
-dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad:02dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad/Assign2dnn/dnn/hiddenlayer_2/biases/part_0/Adagrad/read:0"/
$dnn/hiddenlayer_2/biases/t_0/Adagrad1 "1
�
.dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad:03dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad/Assign3dnn/dnn/hiddenlayer_3/weights/part_0/Adagrad/read:0"3
%dnn/hiddenlayer_3/weights/t_0/Adagrad1  "1
�
-dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad:02dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad/Assign2dnn/dnn/hiddenlayer_3/biases/part_0/Adagrad/read:0"/
$dnn/hiddenlayer_3/biases/t_0/Adagrad "
�
'dnn/dnn/logits/weights/part_0/Adagrad:0,dnn/dnn/logits/weights/part_0/Adagrad/Assign,dnn/dnn/logits/weights/part_0/Adagrad/read:0",
dnn/logits/weights/t_0/Adagrad  "
�
&dnn/dnn/logits/biases/part_0/Adagrad:0+dnn/dnn/logits/biases/part_0/Adagrad/Assign+dnn/dnn/logits/biases/part_0/Adagrad/read:0"(
dnn/logits/biases/t_0/Adagrad ""�
dnn�
�
"dnn/hiddenlayer_0/weights/part_0:0
!dnn/hiddenlayer_0/biases/part_0:0
"dnn/hiddenlayer_1/weights/part_0:0
!dnn/hiddenlayer_1/biases/part_0:0
"dnn/hiddenlayer_2/weights/part_0:0
!dnn/hiddenlayer_2/biases/part_0:0
"dnn/hiddenlayer_3/weights/part_0:0
!dnn/hiddenlayer_3/biases/part_0:0
dnn/logits/weights/part_0:0
dnn/logits/biases/part_0:0"�
	summaries�
�
+dnn/hiddenlayer_0_fraction_of_zero_values:0
dnn/hiddenlayer_0_activation:0
+dnn/hiddenlayer_1_fraction_of_zero_values:0
dnn/hiddenlayer_1_activation:0
+dnn/hiddenlayer_2_fraction_of_zero_values:0
dnn/hiddenlayer_2_activation:0
+dnn/hiddenlayer_3_fraction_of_zero_values:0
dnn/hiddenlayer_3_activation:0
$dnn/logits_fraction_of_zero_values:0
dnn/logits_activation:0
training_loss/ScalarSummary:0""
train_op

train_op/dnn/train"A
local_variables.
,
metrics/mean/total:0
metrics/mean/count:0"&

summary_op

Merge/MergeSummary:0"�
model_variables�
�
"dnn/hiddenlayer_0/weights/part_0:0
!dnn/hiddenlayer_0/biases/part_0:0
"dnn/hiddenlayer_1/weights/part_0:0
!dnn/hiddenlayer_1/biases/part_0:0
"dnn/hiddenlayer_2/weights/part_0:0
!dnn/hiddenlayer_2/biases/part_0:0
"dnn/hiddenlayer_3/weights/part_0:0
!dnn/hiddenlayer_3/biases/part_0:0
dnn/logits/weights/part_0:0
dnn/logits/biases/part_0:0"J
savers@>
<
save/Const:0save/Identity:0save/restore_all (5 @F8"
ready_op


concat:0"U
ready_for_local_init_op:
8
6report_uninitialized_variables_1/boolean_mask/Gather:0"
init_op


group_deps���*       ����	����1�A:reg_large_0314/model.ckpt���B       mS+		V���1�A:�,>QD      xE��	���1�A*��
0
)dnn/hiddenlayer_0_fraction_of_zero_valuesq��=
�
dnn/hiddenlayer_0_activation*�   ���@   ���A!�j�7�١A)�M��(A�A2�        �-���q=d�V�_>w&���qa>:�AC)8g>ڿ�ɓ�i>w`f���n>ہkVl�p>�H5�8�t>�i����v>E'�/��x>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@�������:�           z}ѱA               @              0@              @              @      @              (@       @      (@       @               @      (@               @      0@      @      4@      @@      (@      0@      @       @      <@      4@      <@      <@      D@      D@      J@      U@      L@      N@      P@      R@      \@      U@      R@      Z@      W@      a@      [@      e@     �e@      e@     �q@     �p@      h@     �l@     �q@     �s@     �w@     �w@      y@     �z@      @     ��@      �@      �@     @�@     ��@     ��@     ��@     ��@     ��@     ��@     �@     �@      �@     �@     ��@     P�@     Ф@     ��@     ��@     Ы@     ��@     �@     ��@     �@     ��@     P�@     $�@     �@     ��@     ��@     ^�@     ��@     ��@     ��@     ��@     B�@     L�@     ��@     L�@     ��@    ���@     ��@     ��@    ���@    ���@    ���@     ��@    @��@    @��@    ��@    ���@    �}�@    `��@     �@    ���@    �Y�@    ���@    �~ A    `�A    ��A    @�A    ��A    pW
A    0A    ��A    `tA    �*A    �AA    hDA    �A    �&A    �A    \!A    T�"A    d�$A    �&A    ��(A    `�+A    �<.A    R�0A    *C2A    x4A    � 6A    *=8A    $�:A    �J=A    !@A    Q�AA    V�CA    jEA    �pGA    ��IA    BLA    OA    �QA    ��RA    .TTA   ��5VA    Q=XA    �sZA   �U�\A   �&._A    ��`A   @bA   ��ncA    ��dA   ��eA   @� gA   ��hA   �?�hA    L[iA   �7siA   @TiA   ��6hA    ��fA    h�dA    ��bA    (`A   ��ZA    �
UA    v�OA    ��FA    N`>A    >3A    x�&A    ��A    �!A    ��@    �u�@     ��@     ��@     ��@     P�@      z@      F@        
0
)dnn/hiddenlayer_1_fraction_of_zero_valuesq��=
�
dnn/hiddenlayer_1_activation*�   ��@   ���A!��T�K��A)�0���9�A2�        �-���q=�
L�v�Q>H��'ϱS>��x��U>��u}��\>d�V�_>w&���qa>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�i����v>E'�/��x>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@�������:�           ,G��A              @      @               @      @              @      @               @      @              (@              0@      @      (@              0@       @       @      0@       @      (@      @@      0@      0@      0@      @@      <@      H@      8@      F@      L@      D@      N@      <@      _@      T@      P@      Y@      V@      _@      Y@      X@      V@     �a@      i@      h@      i@      e@     �i@      t@     �s@      p@      x@      v@     �x@     �x@     @~@      |@     `�@     `�@     ��@      �@     ��@     ��@     `�@     Б@     @�@     �@     �@     ��@     ��@     ��@     ��@     8�@     ��@     p�@     �@      �@     8�@     خ@     �@     �@     X�@     t�@     h�@     b�@     ؿ@     ��@     �@     ~�@     ��@     ��@     L�@     ��@     ��@     P�@     �@     ��@     ��@     B�@     
�@     ��@    ���@    ���@    �O�@    @��@     _�@     ��@     �@    @��@    ���@    `E�@    ``�@    ���@    �{�@    ���@    � A     �A    0�A    A    �	A    P�A     FA    �A     wA    8UA    �A    �hA    @�A    @�A    �\ A    ��!A    �#A    ԧ%A    ��'A    �=*A    L�,A    T�/A    �e1A    53A    �5A    F27A    m9A    F<A    (�>A    *�@A    ��BA    �zDA    2{FA    4�HA    A"KA    F�MA    �LPA   �J�QA   ��SA   �/nUA   ��gWA   �t�YA    ��[A    �?^A   �WS`A   @��aA   @6�bA    �[dA   @�eA   @��fA   ��hA   @��hA   �ݥiA   �0�iA   @�iA   @iHiA   ��-hA   ���fA   @hdA   @�aA    (*^A    �`XA    ��RA    I�KA    ��BA    �Z8A    -A    ( A    �A    ���@    ��@     L�@     T�@     ��@     `�@      V@       @        
0
)dnn/hiddenlayer_2_fraction_of_zero_values�/=>
�
dnn/hiddenlayer_2_activation*�   �v�@    ���A!�S}5���A)�6��%�sA2�        �-���q=����W_>>p��Dp�@>������M>28���FP>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�           8�O�A              @               @              @              @              @      @      @      @      @              @       @      @       @      @       @       @       @      @      (@      (@      @      (@      4@      @      4@      8@      0@      8@      B@      B@      4@      8@      D@      8@      0@      S@      H@      J@      L@      N@      U@      J@      V@      Z@      ^@      Y@      b@      d@      f@     �g@     �c@      o@      j@     �u@      p@     �t@     @z@     �v@     �|@     ��@     @�@      �@     ��@      �@     ��@     ��@     ��@     P�@     ��@     ��@     �@     P�@     P�@     p�@     p�@     x�@     8�@     ؤ@      �@     ت@     ��@     `�@     ı@     ��@     ʵ@     h�@     ȹ@     ̼@     ��@     &�@     ��@     ��@     H�@     z�@     ��@     b�@    �m�@     ��@     9�@     ��@     H�@     ��@     ��@     ��@    ��@    @Y�@      �@    @\�@    �h�@    ��@     o�@     ��@     d�@     (�@    `D�@    ���@    �n�@    @��@    �'A    P�A    ��A     �A    @!	A    ��A    0^A    ��A    ��A    �6A    H@A    �vA    � A    ��A    �k A    8�!A    (�#A    x�%A    ��'A    �**A    x�,A    D�/A    �f1A    �B3A    ,&5A    B,7A    �q9A    �	<A    |�>A    ��@A    �BA    MMDA    v@FA    �cHA    ��JA    	/MA    ;�OA   ��:QA   ���RA   �y:TA    ��UA   ��xWA   ��YA   �J�ZA   �J\A   ��H]A   �ZO^A   ��^A    �_A    V�^A    W�]A   ��I\A    |ZA    ;jWA    �FTA    ��PA    �7KA    ��DA    2�>A    �g5A    ��+A    p� A    RA    `9A     ��@    �t�@     :�@     H�@     �@     @|@     �c@      B@      (@        
0
)dnn/hiddenlayer_3_fraction_of_zero_valuesg�>
�
dnn/hiddenlayer_3_activation*�   `�O@    ��A!�˅�� sA)c`ߕ�?YA2�        �-���q=6��>?�J>������M>4�j�6Z>��u}��\>w&���qa>�����0c>cR�k�e>ڿ�ɓ�i>=�.^ol>ہkVl�p>BvŐ�r>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�������:�           �LZ�A              @              @               @      @               @              @              @               @      (@              @      @      @       @      (@      8@              (@      (@      8@      <@      @      4@      <@      (@      4@      0@      4@      D@      8@      F@      8@      N@      N@      H@      <@      D@      ^@      T@      W@      W@      T@     �c@      b@     �`@      e@     �g@     �j@     �i@     �h@      o@      o@      q@      x@     @x@     �v@     @�@     @�@     ��@     ��@      �@     `�@      �@     ��@     @�@     ��@     ��@     ��@     Е@     0�@     @�@     ��@     ��@     ��@     ��@     ��@     �@     ��@     ��@     ��@     �@     ��@     ��@     x�@     �@     (�@     H�@     d�@     >�@     ��@     ��@     ��@     >�@     ��@     �@     ��@     ��@     c�@     ��@    ���@     ��@     ��@    ���@    �e�@    @k�@     ^�@    ���@     �@    �!�@    ���@    ���@     *�@     ��@    ���@    ���@    �;�@    �4A    �A    ��A    PFA    �K	A    P A    p�A    P�A    (�A    h�A    �yA    X�A    �sA    HA    � A    �"A    ��#A    �&A    x,(A    �*A    L]-A    �"0A    �1A    �3A    j�5A    �7A    ��9A    �M<A    ��>A    ��@A    �oBA    �DA    �EA    �pGA    @FIA    SKA    f�LA    �FNA    ��OA   �hYPA   ��PA    �PA    p�PA    �CPA    ��NA    -�LA    �"JA    ��FA    WwCA    J�?A    ny8A    `�1A    ��(A    $ A    ��A     NA    @{�@    ���@     ��@     6�@     ̵@     ��@     ��@      e@      B@       @        
)
"dnn/logits_fraction_of_zero_values    
� 
dnn/logits_activation*� 	   ����   ��/�?    @G\A![�ڟ�:;A)����.!A2�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾄiD*L�پ�_�T�l׾jqs&\�ѾK+�E��Ͼ0�6�/n�>5�"�g��>�*��ڽ>�[�=�k�>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�              4@      L@      ^@      [@      m@     �m@     @w@     ��@     ��@     ��@     `�@     Б@     ��@     ��@     p�@     ��@     (�@     ��@     (�@     �@     ��@     4�@     D�@     ��@     X�@     ��@     |�@     @�@     �@     ��@     �@     ��@     P�@     �@     �@     Ы@     x�@     l�@     �@     �@     �@     С@     ��@     ��@     0�@     ��@     ��@      �@     @�@     ��@     P�@      �@     p�@      �@     @�@     ��@      �@      �@     @�@     ��@     �y@     �|@     ��@     �w@     �r@     �q@      s@     �k@     �n@     �k@     �i@      a@     �`@      W@      [@      X@      [@      ^@      S@      Q@      P@      Y@      B@      F@      J@      J@      B@      <@      0@      B@      8@      D@      J@      <@      (@      8@      @      8@       @       @      @      0@       @      @      0@      @              @      @       @      0@      @               @      <@      @      @              @              @              @              @              @      @              @              @              @               @              0@               @      0@              (@      (@       @      (@      @      8@      <@      @      8@      0@      8@      (@      8@      (@      D@      0@      8@      <@      H@      P@      (@      B@      P@      N@      J@      R@      b@     �`@      W@      Z@      \@     �d@     �i@     �a@     �m@     �j@     �s@     �r@      w@     �w@     �u@     �w@      z@     �@     �@     ��@     ��@     ��@     ��@      �@     ��@     �@     P�@      �@     ��@     �@     ��@     М@      �@     ��@     �@      �@     p�@      �@     ,�@     �@     0�@     (�@     4�@     ,�@     P�@     n�@     t�@     ��@     �@     �@     M�@     ��@     9�@    ���@    ��@    ���@    ���@    ���@    ���@     7�@    �� A    �3A    �NA    �A    �pA    @�A    ��A    P�A    0�A    8�A    �? A    �" A    H�A    ȫA    0�A    �A    ��A    @�A     \�@    @E�@     ��@     @�@     ��@     X�@     ��@     �q@      \@      D@        

loss[Ŀ?�_]*       ����	���1�A:reg_large_0314/model.ckpt��;*       ����	SvM'�1�A8:reg_large_0314/model.ckpt<1�?*       ����	X7⾯1�AT:reg_large_0314/model.ckpt8ysr%       �6�	�
{�1�Ae*

global_step/sec�d:=���F5      5~��	N{�1�Ae*�j
0
)dnn/hiddenlayer_0_fraction_of_zero_valuesq��=
�
dnn/hiddenlayer_0_activation*�   ��	@   ���A!�c�?��A)I��ܓ�A2�        �-���q=_"s�$1>6NK��2>�so쩾4>/�p`B>�`�}6D>H��'ϱS>��x��U>d�V�_>w&���qa>cR�k�e>:�AC)8g>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@�������:�           ����A              @      @               @              @              @              @              @      @       @              @       @              @       @              (@       @      0@      @      @      $@      0@      (@      (@      4@      0@      B@      0@      0@      @@      <@      B@      F@      8@      F@      D@      J@      T@      L@      V@      k@      W@      V@      [@      ^@      ^@     �d@      X@      i@      h@      m@     �m@     �l@     �s@     �p@     @v@     �w@     �y@      |@     �|@      �@     @�@     ��@     ��@     `�@      �@      �@     ��@     @�@     0�@     ��@     ��@     ��@     @�@     $�@     �@     ��@     X�@     P�@     0�@     ȫ@     X�@     p�@     �@     |�@     ��@     ĸ@     h�@     ��@     J�@     �@     ��@     T�@     W�@     
�@     @�@     \�@     #�@    �u�@     K�@     �@     ��@     ]�@     �@    ��@    ��@    @��@    �d�@    ���@     ��@    �8�@    @I�@    ``�@    �d�@     m�@    ���@    �O�@    @�@    �� A    ��A    0�A    0�A    0�A     J
A    �A    ��A    �A    �TA    �<A    �LA    ��A    �IA    ��A    �!A    ܬ"A    p�$A    ��&A    $�(A    (V+A    T:.A    ��0A    �B2A    .4A    �
6A    BI8A    8�:A    $f=A    @A    k�AA    k|CA    �iEA    �|GA    ��IA    �LLA    )
OA    �QA   �ʬRA    }dTA   �HVA   ��MXA   �E}ZA    ��\A   ��6_A   @�`A    11bA    ��cA   ���dA   �fA   � MgA    OBhA   ��iA    A�iA    �iA   @�hiA   @�hA   �~1gA   ��PeA   ��cA   �^`A   �#[A   �upUA    �)PA    ;GA    D�>A    �l3A    ��&A    �-A    �.
A    �	�@     ?�@     ��@     v�@     ��@      �@      h@      (@        
0
)dnn/hiddenlayer_1_fraction_of_zero_valuesq��=
�
dnn/hiddenlayer_1_activation*�    @   ���A!V$i�Z�A)s(�߄�A2�        �-���q=�f׽r��=nx6�X� >p��Dp�@>/�p`B>��x��U>Fixі�W>4�j�6Z>��u}��\>w&���qa>�����0c>cR�k�e>:�AC)8g>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@�������:�           ��ɱA              @              @              @       @      @              @      (@      @              4@      @      @      @      (@              @               @              @      @      (@      0@      (@      8@      @@      0@      8@      4@       @      <@      B@      B@      0@      <@      @@      H@      <@      B@      H@      N@      L@      X@      J@      U@      U@      W@      _@     �a@     �f@     �c@      e@     �r@      n@      s@     �p@     �r@     @r@      x@     @u@     �~@     @~@      �@     ��@      �@     @�@     ��@      �@     ��@     �@     ��@      �@     ��@     P�@      �@     К@     ��@     (�@     ��@     ��@     X�@     ��@     ج@     ��@     ��@     4�@     ��@     D�@     ��@     �@     &�@     |�@     ��@     $�@     B�@     ��@     ��@     j�@     >�@     ��@     7�@     ��@     -�@     ��@     �@     ��@     \�@    �4�@    ���@     3�@     �@     I�@     1�@    ��@    @��@    @��@    ��@    @�@    ���@    ���@    Н A    3A    �A    `�A    �\A     �
A    �tA    `NA    ��A    ��A    �A    `zA    �AA    X�A    ��A    S!A    �#A    h�$A    �>'A    ,C)A    d�+A    ��.A    ��0A    p�2A    \o4A    ~6A    ��8A    �@;A    D�=A    ct@A    �BA    ��CA    *�EA    C�GA    �KJA    {�LA    b�OA   ��YQA   �p�RA   �/�TA   �b�VA    ]�XA    ��ZA    �E]A   ���_A    y*aA    ԀbA    ��cA    q(eA   �ifA   @�gA    ��hA   ��UiA    x�iA   ���iA    �OiA   �AQhA   ���fA   ��dA   �CbA    ��^A    �YA    bSA    8ZLA    �zCA    l 9A    t�-A    ��A    `A     ��@    �Q�@     ��@     l�@     ��@     �p@      B@       @        
0
)dnn/hiddenlayer_2_fraction_of_zero_values�/=>
�
dnn/hiddenlayer_2_activation*�   ���@    ���A!�R�h�\�A)Q(�d�tA2�        �-���q=������M>28���FP>�����0c>cR�k�e>ڿ�ɓ�i>=�.^ol>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�           ��3�A               @              (@              @              (@      0@      (@              @              @       @      @              8@      (@       @      0@              @       @      @       @      <@      4@      D@       @      4@      @@      B@      <@      @@      @@      D@      L@      N@      V@      Q@      S@      W@      a@     �`@     �`@     �j@     �`@      _@      k@      k@     �k@      m@     �v@     �r@     @s@     �u@      ~@     �{@     ��@      �@     `�@     `�@     ��@     ��@      �@      �@     P�@     `�@     `�@     �@     �@     @�@     ��@     p�@     0�@     �@     0�@     ��@     ��@     ��@      �@     ذ@     ��@     x�@     P�@     �@     �@     ��@     ��@     �@     ��@     t�@     B�@     t�@     ��@     ^�@     K�@     �@     >�@     d�@     )�@     v�@     \�@    ��@    ��@    ���@     ��@     ��@    @��@     ��@    ��@    �z�@    �j�@    �.�@     o�@    ���@    �z�@    �t A    ��A    0�A    ��A    ��A    0,
A     �A    �A    �iA    hQA    03A    �>A    �A    �=A    P�A    �!A    �"A    H�$A    �&A    ��(A    �B+A    �.A    ��0A    8.2A    p�3A    ~�5A    �%8A    ��:A    �-=A    �@A    ��AA    �JCA    �)EA    �#GA    �gIA    J�KA    �SNA    ڈPA    �RA   �ǚSA   �CUA    ��VA   ���XA   ��jZA   ���[A   ��]]A   ��^A    �O_A    ��_A    �5_A   ��^A    �,\A   �R�YA    �QVA   �5�RA    d�MA    ��FA    �n@A    P6A    5,A    d� A    h	A    0A     �@     z�@     ��@     Ч@     ��@     �i@      F@      @        
0
)dnn/hiddenlayer_3_fraction_of_zero_valuesg�>
�
dnn/hiddenlayer_3_activation*�   ����?    ��A!�fX�h�|A)ު{xPfA2�        �-���q=p
T~�;>����W_>>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>K���7�>u��6
�>T�L<�>��z!�?�>�4[_>��>
�}���>X$�z�>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�           �$��A              @               @              (@              @               @              @              @      @              (@       @      @      (@      0@       @      @      8@      (@      0@      0@      4@      @       @      8@      @@      D@      L@      4@      J@      @@      J@      H@      R@      S@      R@      N@      [@      `@      _@      d@     �b@      f@     �b@      c@      o@      p@     �m@     �u@     �v@     @w@     @@      {@      �@      �@     ��@     `�@     ��@     ��@     ��@     ��@      �@     ��@     ��@     `�@     ��@      �@     �@     ��@      �@     `�@     x�@     ��@     �@     `�@     @�@     �@     ��@     ̳@     (�@     ��@     �@     ��@     ��@     j�@     .�@     ��@     &�@     �@     ��@     ��@     
�@     �@     \�@     ��@     ��@     ��@     b�@    @K�@     s�@    �M�@    @^�@    ��@    �;�@     �@    ���@    ���@     ��@    �j�@     ��@    `��@     s�@    �_�@     ZA    `�A    ��A    ��A    p+	A    �A    ��A    P�A    �lA    �6A    �QA    �RA    �A    �kA    t$ A    ��!A    �#A    �%A    Ȕ'A    �)A    L�,A    PZ/A    �L1A    �3A    �5A    �7A    |�9A    �-<A    �X?A    �_AA    �aCA    R�EA    5\HA    �sKA    �OA   ��mQA   ���SA   ���UA   �F�WA    �QYA   �E1ZA   ��	ZA    ��XA   �� VA    �VRA    �hLA    �BDA    x:A    ,X/A    x� A    p�A    ���@    ���@     ��@     0�@     �@      w@      H@      @      @        
)
"dnn/logits_fraction_of_zero_values    
�
dnn/logits_activation*�	   �+��?   `��?    @G\A!@ӌ�7�[A)+�W|[A2��QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�������:�              @              @      @               @       @      <@      J@      {@     `�@     ��@     P�@     �@     !A    <�%A    (6A    ��>A    D�9A    �)A    @�A    ���@     ��@     `�@      T@        

loss��k?Ll��*       ����	(��X�1�Ap:reg_large_0314/model.ckpt%��+       ��K	��E�1�A�:reg_large_0314/model.ckptX�~�+       ��K	�+B��1�A�:reg_large_0314/model.ckpt��+       ��K	��3!�1�A�:reg_large_0314/model.ckpt�N��&       sO� 	p"u>�1�A�*

global_step/sec��;=W@dg5      �Ѵ	�yu>�1�A�*�j
0
)dnn/hiddenlayer_0_fraction_of_zero_valuesq��=
�
dnn/hiddenlayer_0_activation*�    �@   ���A!��<�"�A)���!��A2�        �-���q=ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>�H5�8�t>�i����v>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@�������:�           ����A              @              @              @               @               @      (@      @      @       @      @      0@      @       @      @      4@      4@       @      8@      0@      <@      4@      <@      <@      J@      8@      D@      S@      F@      F@      R@      P@      V@      U@      \@      V@      R@     �`@      _@      h@      `@      _@      f@     �j@     �m@      m@      n@      l@     @w@     �r@     �x@     @y@     @@     �|@     �}@     ��@     ��@     ��@     @�@     ��@     ��@     �@      �@     p�@     ��@     ��@     @�@      �@     8�@     ��@     0�@     P�@      �@     ��@     h�@      �@     \�@     �@     ��@     |�@     �@     ��@     8�@     ��@     
�@     ��@     ��@     ��@     K�@     ��@     �@     w�@     �@    �b�@     '�@     Y�@     |�@    ���@     l�@     �@    ���@    ���@     (�@    ��@     i�@    `��@    �4�@    @;�@    �_�@    �y�@     �@    @��@    ] A    �A    �A    ��A    ��A    �
A    ��A    @�A    ЏA    ZA    8A    �nA    x�A    `QA    �A    4!A    ��"A    ��$A    ��&A    �(A    �}+A    ;.A    ��0A    �<2A    4A    6A    W8A    |�:A    �a=A    &@A    >�AA    OzCA    �qEA    2}GA    ��IA     HLA    TOA   ��QA    �RA    �hTA   ��WVA    �TXA   �W�ZA    }�\A    \8_A   ���`A   ��6bA   �e�cA    s�dA   �^#fA   ��YgA   @�ShA   �.!iA   @K�iA   ���iA   ��{iA   �x�hA   �%OgA   ��neA   @�%cA   ��{`A   ��K[A    :�UA    �IPA    h.GA    ,?A    �3A     �&A    �NA    �
A    @��@      �@     [�@     ��@     ��@     ��@      h@      (@        
0
)dnn/hiddenlayer_1_fraction_of_zero_valuesq��=
�
dnn/hiddenlayer_1_activation*�   @K�@   ���A!V6��ih�A)�jX��V�A2�        �-���q=%���>��-�z�!>��Ő�;F>��8"uH>d�V�_>w&���qa>cR�k�e>:�AC)8g>ہkVl�p>BvŐ�r>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@�������:�           ^���A              @              @              (@              @               @              (@      (@       @       @      (@      @      @      4@      @      0@      @      4@      4@      @@      8@      4@      0@      <@      (@      0@      8@      H@      J@      F@      [@      L@      N@      R@      Q@      X@      T@      V@      Y@     �c@      e@      `@     �e@      b@     �e@      j@      m@     �l@      s@     @r@     @r@     @u@     �x@     ��@      ~@     ��@     `�@      �@      �@     @�@     ��@     ��@      �@     �@     ��@      �@     0�@     @�@     �@     8�@     p�@     `�@     ��@     `�@     إ@     ��@     د@     �@     L�@     �@      �@     D�@     ��@     ��@     ��@     �@     ��@     ��@     ��@     ��@     N�@     ��@     l�@     p�@     K�@      �@     ��@    ���@    �	�@    @��@    �)�@    @�@    @��@    �2�@    ��@    �\�@    ���@    �v�@     r�@    ���@    ���@    `��@     ��@    pO A    �A    ��A    ��A     A    ��
A     6A    8 A    8�A    �[A    �rA    H�A    ��A    �oA    �7A    h2!A    l�"A    (�$A    ��&A    �,)A    @�+A    �T.A    ��0A    �M2A    @C4A    �B6A    �{8A    ��:A    ԓ=A    �<@A    s�AA    �CA    �EA    ��GA    � JA    �LA    �KOA    w&QA    �RA    ��TA   �2qVA   �K|XA   ���ZA   ��\A   �Kx_A   �
aA   �yUbA   @G�cA   ��eA   @�IfA   �ZygA   �^|hA    �MiA   ���iA    9�iA   �iviA   �^�hA    $gA   @eeA    �bA    �_A   ���YA   �\*TA    Q�MA    ��DA    p�:A    ��/A    �=!A    `�A    ���@    ���@     ,�@     ��@     ��@     �q@      F@       @        
0
)dnn/hiddenlayer_2_fraction_of_zero_values�/=>
�
dnn/hiddenlayer_2_activation*�   @q@    ���A!no��.�A)�c��uA2�        �-���q=���">Z�TA[�>��R���>Łt�=	>����W_>>p��Dp�@>������M>28���FP>��u}��\>d�V�_>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�           �ԥA              @              @              @              @              @              @              @               @              @      @       @      @      (@       @      0@      @      0@       @      4@      (@      0@      4@       @      @       @      0@      0@      0@      <@      8@       @      B@      F@      <@      J@      <@      F@      J@      T@      R@      W@      [@      Y@      S@      X@      d@     �a@      b@     �d@      g@      i@     �l@     �r@     �q@     �s@     @r@     �t@     @v@      w@      |@     ��@     ��@     @�@     `�@      �@     ��@     ��@     ��@      �@     В@     ��@     ��@     З@     К@     ��@     @�@     (�@     0�@     X�@     P�@     ��@     �@     �@     ��@     X�@     4�@     �@     0�@     d�@     ��@     ��@     ��@     ��@     ��@     (�@     ��@     �@     [�@     ��@     ��@    �q�@     m�@     ��@     ��@    �4�@    @R�@    ���@    ���@    @��@     ��@     }�@    �d�@    �0�@    @	�@    ���@    ��@     $�@    ���@    ���@    �~A    �GA    �jA    `�A    ��	A    @�A    �A    ��A    ��A     �A    p�A    `EA    �A    �mA    \� A    �f"A    �>$A    �C&A    ��(A    ��*A    İ-A    fF0A    ��1A    R�3A    �5A    ��7A    �%:A    d�<A    ��?A    iAA    M#CA    �
EA    DGA    ZBIA    ��KA    �INA   �؍PA   �YRA   ���SA    �\UA   �ZWA   ���XA    L�ZA   ��h\A   ���]A   ��=_A   �G`A   ��8`A    �`A    �_A    >G]A   �p�ZA   ��`WA   �~�SA    u�OA    K$HA    r�AA    d�7A    @[.A    p�!A    �A    P�A    ���@     ��@     j�@     h�@      �@     �o@      8@        
0
)dnn/hiddenlayer_3_fraction_of_zero_valuesg�>
�
dnn/hiddenlayer_3_activation*�    �_�?    ��A!զhk�<}A)��O�L�fA2�        �-���q=�z��6>u 5�9>/�p`B>�`�}6D>4�j�6Z>��u}��\>w&���qa>�����0c>:�AC)8g>ڿ�ɓ�i>f^��`{>�����~>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�������:�           ����A              @              @              @              @              @              @               @      @      @      @       @      @              @               @               @       @       @       @      0@      (@      (@              4@      0@      8@      0@      <@      @@      P@       @      <@      D@      N@      H@      P@      J@      S@      T@      P@      T@      S@      Q@      V@      b@      _@     �d@     �a@     �i@      g@      n@      j@     �p@      m@      v@     �t@     �w@     @{@      �@     ��@     ��@     ��@     @�@     ��@     ��@      �@      �@     @�@     �@      �@     @�@     И@     `�@     0�@     H�@     ̡@     �@     h�@     ا@      �@     ج@     H�@     �@     �@     Դ@     H�@     H�@     ��@     ��@     �@     s�@     ��@     ~�@     T�@     b�@     ��@     �@    �;�@     �@    ���@     !�@     ��@    ���@     ��@     ]�@     ��@    ���@    @��@    @b�@     "�@     ��@    @��@     ��@    ���@    ���@    `Z�@     ��@    ��@     � A    @TA    �1A    p_A    �lA    0�
A    ��A    �=A    P�A    ��A    x�A    ��A    �1A    0�A    P�A    �_!A    �#A    ��$A    �'A    @a)A    ��+A    ��.A    ��0A    v�2A    |4A    ��6A    �9A    ^�;A    ��>A    -5AA    L@CA    `�EA    �1HA    |KKA    ��NA   �vQQA    �rSA   ���UA   ��WA   ��-YA   ��4ZA   ��AZA    FYA    !rVA    N�RA    �{MA    xEA    ��;A    �`0A    ��!A    `fA     V�@     N�@     ��@     ��@     0�@     @{@      ]@      8@        
)
"dnn/logits_fraction_of_zero_values    
�
dnn/logits_activation*�	   `(�?    �� @    @G\A!@��Nǧ[A)�����[A2�!�����?Ӗ8��s�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�              @              @      0@      @      0@      @@     �h@     ��@     ��@     $�@     ��@     ��@    @��@    ��A    ��A    �H&A    ,d2A    ا8A    HU8A    �(/A    8�A    ���@     ��@     ��@     �j@      <@        

loss��h?+��+       ��K	�'���1�A�:reg_large_0314/model.ckpt�!	+       ��K	7 DT�1�A�:reg_large_0314/model.ckpt�b��+       ��K	�EH�1�A�:reg_large_0314/model.ckpt[&       sO� 	��^�1�A�*

global_step/sec�/<=3t�1�5      �GS	�B�^�1�A�*�j
0
)dnn/hiddenlayer_0_fraction_of_zero_valuesq��=
�
dnn/hiddenlayer_0_activation*�    �!@   ���A!N���j0�A)V(%�֝A2�        �-���q=Fixі�W>4�j�6Z>:�AC)8g>ڿ�ɓ�i>w`f���n>ہkVl�p>BvŐ�r>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@�������:�           |��A              @               @              @      @               @      (@      @      @      @       @      (@      (@       @      @      @      8@      (@      4@      (@      (@      0@      8@      (@      <@      D@      4@      4@      @@      D@      4@      N@      4@      L@      J@      F@      W@      R@      W@      P@      \@      c@      _@     �b@      `@      h@      d@     �g@      m@      n@     �s@      t@     �p@     @w@     �y@     @y@     @x@     �@     ��@     ��@     @�@     `�@     ��@      �@     0�@      �@     ��@     `�@     ��@     И@     ��@     ��@     h�@     @�@     ئ@     ܧ@     p�@     ��@     �@     ��@     �@     0�@     ��@     P�@     Լ@     Ƚ@     �@     ~�@     &�@     ^�@     r�@     ��@     ��@     0�@     ��@     Q�@     ��@     ��@     ��@     Q�@     ��@    @�@     ��@    �I�@     ��@    ���@    ���@    ���@    ���@    ���@     ��@    �]�@    ���@    @��@    ��@    `A A    �A    p�A    ��A    ��A    �
A    ��A    0�A    �]A    HA     *A    �?A    ��A    �kA    �A    �!A    м"A    4�$A    ؾ&A    �)A    `^+A     :.A    ��0A    x;2A    �4A    �6A    tC8A    �:A    ,R=A    2 @A    ��AA    �zCA    ipEA    ��GA    ��IA    �ULA    YOA   �AQA    �RA    srTA   ��QVA   �bXA    }�ZA   ���\A   �ED_A   @`�`A    W8bA    ��cA   @�dA   ��1fA   @fVgA    jhA    �*iA   �'�iA   �7�iA    �iA   @�hA   ��cgA   �y�eA    9cA   @Ɏ`A    Aq[A   �ʻUA   ��[PA    WGA    <8?A    ޥ3A    ��&A    �SA     
A    ���@    ���@     �@     d�@     ��@     ��@     �g@      (@        
0
)dnn/hiddenlayer_1_fraction_of_zero_valuesq��=
�
dnn/hiddenlayer_1_activation*�   @�[@   ���A!
'6~�A)����A2�        �-���q=Z�TA[�>�#���j>p
T~�;>����W_>>�
L�v�Q>H��'ϱS>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@�������:�           \M��A              @              @              @              @      @      @              @              @              @      @              @      @              @               @       @      (@      (@      4@      4@       @      8@       @      (@      (@      4@      <@      8@      8@      <@      0@      B@      D@      H@      L@      L@      P@      P@      U@      S@      W@      T@      \@      a@      `@      a@     �i@      g@     �l@     �k@     �p@      n@      q@     �x@      t@     @v@     �{@     �{@     �@      �@     ��@     ��@     ��@     ��@     ��@     0�@      �@     ��@     ��@     p�@     �@     �@     �@      �@     Т@     Ȥ@     Ц@     Ъ@     ��@     ��@     �@     L�@     *�@      �@     |�@     �@     l�@     :�@     �@     ��@     �@     �@     9�@     v�@     ��@     ��@    ���@     ��@     ��@    ���@     A�@     ��@     -�@     ��@     ��@    ���@     �@    �2�@     ��@    `��@    �o�@    ���@     0�@    `��@    ���@    ���@    @� A    @&A    P�A    @�A    `�A    @u
A    ��A    ��A    ��A    0aA    �MA    0cA    ��A    �.A    0)A    P!!A    ��"A    t�$A    �&A    �)A    �+A    `.A    г0A    lN2A    �34A    �J6A    �m8A    "�:A    �s=A    �A@A    ��AA    :�CA    ��EA    ��GA    +�IA    �zLA    )EOA   �ZQA   ��RA    �~TA    �dVA   �2tXA    N�ZA   �7�\A   ��q_A   @aA    �WbA   ���cA   �m eA    �GfA   �^vgA    .�hA   �:IiA    �iA   @
�iA    *iA   ���hA   ��1gA   @;=eA   �&�bA   ��`A    �$ZA    QLTA    i�MA    ��DA    ��:A    ��/A    �Z!A    � A    @��@    ��@     @�@     ��@     ��@     @t@      D@      @        
0
)dnn/hiddenlayer_2_fraction_of_zero_values�/=>
�
dnn/hiddenlayer_2_activation*�   �|�@    ���A!���	��A)�<�w��tA2�        �-���q=������M>28���FP>�
L�v�Q>H��'ϱS>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�           ��A              @              @              @      @              @              @      @               @               @       @      @       @      4@      0@      @      4@      <@      @      8@       @      4@      4@      @      @      @@      4@      <@      <@      B@      B@      <@      B@      D@      R@      T@      Q@      Y@      J@      V@      \@     �f@      _@      f@      c@      d@     �b@      j@      j@     �t@     �n@     @w@      x@      u@      w@      ~@     �{@     ��@     `�@     ��@      �@     `�@     `�@     ��@     А@     ��@      �@     `�@     Ж@      �@     К@     �@     ��@     Т@     X�@     (�@     (�@     ��@      �@     h�@     �@      �@     d�@     ��@     P�@     \�@     ��@     ��@     ��@     ��@     ��@     H�@     �@     ��@     ��@     s�@     %�@     r�@     ;�@     ��@     ��@     C�@    ���@    @��@     �@    ��@     ��@    ���@    `��@    �
�@    �k�@    �Q�@    `��@    ���@    ���@    �Z A    �A     �A    ��A    ��A    0�
A     A    0�A    ГA    �?A    �+A    `vA    �A    [A    ��A    0!A    ��"A    ��$A    ��&A    x)A    ��+A    �;.A    ��0A    ,`2A    34A    %6A    b_8A    j�:A    .�=A    f3@A    ��AA    G�CA    tEA    �GA    F�IA    �XLA    ~�NA    ��PA    (iRA   � TA   �L�UA   �$�WA    �BYA   ���ZA   ��\A   �X^A    �H_A   �J`A   ��`A   �$�_A   �r�^A    &r\A    4�YA   �ZVA   � �RA    :�MA    C}FA    �@A    ��5A    �+A    0�A    `�A     � A    �y�@     ��@     ^�@     ��@     ��@      c@      0@        
0
)dnn/hiddenlayer_3_fraction_of_zero_valuesg�>
�
dnn/hiddenlayer_3_activation*�    �] @    ��A!!�'M�{A)�ؒj eA2�        �-���q=H��'ϱS>��x��U>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�           �}A�A              @              @              @      @              @      @              (@      @      (@               @      0@       @      @      @      @               @       @       @       @              4@      @@      0@      (@      4@      B@      @@      4@      F@      F@      F@      L@      B@      @@      D@      R@      R@      T@      S@      X@      b@      P@      U@     �b@     �d@      c@     �e@     �m@     �j@     �n@      p@     �p@      m@     �y@     @u@     ��@     �{@     @�@      �@     ��@     @�@      �@     ��@     ��@     0�@     ��@     ��@     �@     @�@     p�@     ��@      �@     @�@     ��@     (�@     (�@     ��@     h�@     ��@     t�@     ر@     �@     @�@     �@     ��@     �@     ��@     2�@     i�@     ��@     �@     ��@     7�@     ��@     t�@     �@     ]�@     ��@    ���@     ��@     ��@    �8�@    @��@     �@     ��@    ���@    ���@    ���@    ���@    @��@    @/�@    �P�@    `~�@    ���@     w�@    �9 A    0	A     �A    ��A     �A    �H
A    �A    p�A    (sA    �7A    0:A    �FA    h�A    �A    p�A    � A    �"A    L�$A    ��&A    H�(A    Q+A    .A    ��0A    92A    Z4A    66A    �T8A    ��:A    ��=A    !w@A    K:BA    $PDA    �FA    �FIA    �3LA    �OA    ��QA    �ySA    �PUA   ���VA    �UXA    Z�XA    ��XA    pHWA    Q�TA    �IQA     �JA    ��BA    P\8A    ��,A    �jA    ��A    @X�@    ���@     ~�@     ��@     В@     @s@      \@      F@       @        
)
"dnn/logits_fraction_of_zero_values    
�
dnn/logits_activation*�	    c3�?   �&@    @G\A!@u� �ZA)�K6GHZA2�� l(��?8/�C�ַ?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�              @              (@      @      (@      B@      T@      f@     `�@     ��@     P�@      �@     R�@    ���@    @��@    �GA    �[A    l A    \�)A    ��2A    ��6A    R5A    $P*A    ��A    �\�@     ��@     �@     @r@      <@        

lossr�f?��+       ��K	�0��1�A�:reg_large_0314/model.ckpth��+       ��K	�*t�1�A�:reg_large_0314/model.ckpt�Po�+       ��K	&�!��1�A�:reg_large_0314/model.ckptJi�F+       ��K	)�PK�1�A�:reg_large_0314/model.ckptQ3��&       sO� 	B�|x�1�A�*

global_step/sec�_>=���g6      V��	c'}x�1�A�*�l
0
)dnn/hiddenlayer_0_fraction_of_zero_valuesq��=
�
dnn/hiddenlayer_0_activation*�    �"@   ���A!M��6�?�A)������A2�        �-���q=p
T~�;>����W_>>��Ő�;F>��8"uH>������M>28���FP>Fixі�W>4�j�6Z>�����0c>cR�k�e>ڿ�ɓ�i>=�.^ol>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@�������:�           ����A              (@              @              @              @               @               @              @       @       @      @       @               @      @              @               @      0@       @      (@      @      8@      4@      (@      <@      8@      8@      @@      @@      D@      0@      F@      J@      <@      J@      Q@      L@      R@      Q@      R@      d@      a@      [@      b@      k@     �b@      d@     �i@     �l@      m@     �s@     �q@      l@     `w@     �v@      x@     �|@     �|@     `�@      �@     @�@     @�@     `�@     0�@     ��@     ��@     ��@     �@     p�@      �@     ��@     �@     ��@     �@     @�@     P�@     |�@     �@     x�@     �@     �@     �@     ��@     ��@     �@     6�@     �@     �@     ��@     R�@     ��@     ��@     �@     ��@     ��@    � �@     ��@     q�@     ��@    ���@    ���@     ��@    �m�@    ��@     I�@     ��@     ��@    ���@     {�@    ���@    �n�@    @E�@    �-�@    �l�@    ��@    ��@    �m A    �"A    ��A     A    @�A     D
A     �A    8A    ppA    �4A    HA    @EA    `�A    0%A    �A    $%!A    �"A    ��$A    p�&A    � )A    �l+A    �&.A    @�0A    �@2A    4A    46A    �I8A    ��:A    �M=A    v@A    l�AA    �oCA    YEA    1�GA    ��IA    0cLA    OA    �QA    �RA    rTA    }TVA    @\XA    �ZA    @�\A   ��P_A    '�`A    @=bA   @*�cA   �v�dA   �#7fA    �`gA    	thA   ��5iA    ��iA   ���iA   ���iA   @~�hA   �:ygA   ���eA   ��QcA   �V�`A   �t�[A    ��UA   � vPA    �{GA    Dl?A    ��3A    �'A    �cA    �*
A    �N�@    �k�@     �@     ��@     (�@     �@     �e@      (@        
0
)dnn/hiddenlayer_1_fraction_of_zero_valuesq��=
�
dnn/hiddenlayer_1_activation*�   `��@   ���A!��0ŋ�A)�Q��Ռ�A2�        �-���q=Z�TA[�>�#���j>����W_>>p��Dp�@>H��'ϱS>��x��U>Fixі�W>4�j�6Z>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�������:�           t���A              @              @              @      @      (@              @               @      (@              @      @       @      0@       @              @      @      (@      (@       @      @      (@      (@      (@      <@      (@      4@      D@      8@      4@      F@      F@      <@      @@      8@      D@      (@      P@      S@      F@      R@      U@      B@      `@      X@      V@     �b@     �`@      d@      c@     �j@      h@      l@      o@      m@      u@     @t@     �x@     @z@     �~@      ~@     ��@     `�@     @�@      �@     ��@     ��@     @�@     ��@     ��@     ��@     0�@     ��@     Й@     ��@     ��@     ȡ@     8�@     (�@     ��@     �@     X�@     �@     ,�@     �@     �@     ��@     $�@     ��@     |�@     �@     ��@     4�@     4�@     ��@     2�@     $�@     ��@     �@     6�@    ���@     ��@     �@    ���@    ��@    �_�@    @��@    ���@    �3�@    ��@     ��@    ���@     p�@    `o�@    ���@    `�@    @Y�@    �a�@    ���@     t A    @9A    ��A    ��A    >A    ��
A    �A    ��A    ��A    �DA    �WA    �lA    �A    xnA    �A    L&!A    ��"A    ��$A    ��&A    �4)A    ��+A    �N.A    
�0A     e2A    ~>4A    ,?6A    �q8A    ��:A    ��=A    �E@A    ��AA    F�CA    B�EA    6�GA    ��IA    �LA    �>OA    UQA   ���RA    ؊TA    TsVA    �}XA   ���ZA   ��]A   �S{_A   @aA   ��[bA   ���cA   �XeA   ��PfA   �J�gA    ��hA    �QiA    ��iA   ���iA   �J�iA   ���hA   �.@gA   ��MeA   ���bA   �"`A   �)8ZA    s_TA    "NA    [�DA    ��:A    �/A    B!A     �A     `�@    ���@     ��@     �@     ��@     �v@      F@        
0
)dnn/hiddenlayer_2_fraction_of_zero_values�/=>
�
dnn/hiddenlayer_2_activation*�   �0�@    ���A!r�Q���A)ϛ~J;esA2�        �-���q=7'_��+/>_"s�$1>��8"uH>6��>?�J>������M>28���FP>4�j�6Z>��u}��\>cR�k�e>:�AC)8g>=�.^ol>w`f���n>E'�/��x>f^��`{>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�           8��A              D@              @              @               @              @               @               @              0@      @              (@       @      @      0@      @       @              @      @      8@      @      (@      <@      4@      4@      <@      D@      B@      @@      8@      B@      D@      L@      P@      P@      Y@      Q@      Y@      W@      S@      V@      Y@      ^@      c@     �a@      j@      e@      c@      k@     �p@     �p@     �q@     @u@     �y@     �|@     `�@     ��@     `�@     `�@      �@      �@      �@     ��@     ��@     ��@     @�@     P�@     0�@      �@     0�@     Р@     ��@     ��@     Ц@     P�@     �@     0�@     ��@     �@     �@     t�@     ��@      �@     ,�@     ��@     ��@     ,�@     
�@     ��@     ��@     ��@     �@     U�@     �@     0�@     ��@     ��@     ?�@     ��@    ���@     �@    ���@     ��@     ��@     D�@    ���@     ��@     T�@    �H�@    `��@    ���@    ���@    @��@    �q�@    �� A    ��A    p&A    �+A    �A     A    ИA    hOA    8A    ؤA    ��A     �A    H@A    �A    ��A    �!A    �S#A    �7%A    �B'A    ��)A    X=,A    4/A    �1A    ��2A    x�4A    ��6A    �8A    y;A    � >A    ��@A    t2BA    ~�CA    ��EA    IHA    {`JA    ��LA    kqOA   ��2QA    ��RA   �#HTA    |VA   ���WA   �`cYA   ��[A    R�\A    ��]A    f_A    �_A   �ɫ_A    � _A   �l�]A    �l[A   �_�XA    �.UA   ���QA    R�KA    �DA    �>=A    �F3A    X�'A    (rA    p#A     )�@    @6�@     ��@     P�@     ��@     ��@      [@       @        
0
)dnn/hiddenlayer_3_fraction_of_zero_valuesg�>
�
dnn/hiddenlayer_3_activation*�    �� @    ��A!7���d�zA)�DŵcA2�        �-���q=������M>28���FP>4�j�6Z>��u}��\>d�V�_>w&���qa>=�.^ol>w`f���n>�H5�8�t>�i����v>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�            ӏ�A              @              @              @               @              @               @      @      @      @               @      @      @       @      @      (@       @      @      (@      (@      @              4@      4@      0@       @      (@      @@      8@       @      4@      <@      <@       @      0@      @@      F@      B@      R@      L@      @@      H@      Q@      ]@      T@      W@      _@      Y@      `@      d@     �b@     �e@      g@      k@     �m@      n@      q@      r@      u@     �u@     �z@     @@     �~@     @�@      �@     ��@     `�@     ��@      �@     ��@     �@     0�@     ��@     З@     ��@     0�@     �@     ��@     Т@      �@     �@     �@     �@     H�@     ȯ@     \�@     ,�@     Ե@      �@     0�@     ��@     �@     ��@     ��@     ��@     b�@     ��@     H�@     ��@     ��@     4�@     ?�@     ��@     �@     =�@     ��@    @�@     ��@    ���@    ���@     V�@    �a�@    �#�@    �}�@     �@    @��@     ��@    �7�@    �T�@    `��@    ��@    ��A    0kA    @A    ��A    ��	A    P�A    CA    �A    H�A    ��A    �A    �A    �A     WA    �� A    p>"A    `$$A    �&A    (E(A    T�*A    HH-A    �0A    �1A    tz3A    �^5A    g7A    f�9A    hu<A    �V?A    )BAA    UCA    EA    �]GA    ��IA    ��LA    ��OA   �&QA   �9SA   ��TA    �LVA    �bWA    ��WA    �^WA    ��UA   �pUSA    h�OA    �xHA    m#AA    Ҷ5A    (&)A    0 A    �kA     ��@     G�@     ��@     8�@      �@     @r@      R@      F@      @@        
)
"dnn/logits_fraction_of_zero_values    
�
dnn/logits_activation*�	   ���?   �@    @G\A!`ʝnuZA)i0Z��YA2��{ �ǳ�?� l(��?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�������:�              @              @       @      4@      @      (@      U@      b@     �r@     ��@     `�@     ��@     h�@     �@     ��@    �P�@    �U�@    �V	A    ��A     �!A    <*A    f�1A    65A    �x3A    �Q)A    ��A    `��@     r�@     ��@     �w@      W@      8@        

loss��d?��T+       ��K	G���1�A�:reg_large_0314/model.ckpt"�X�+       ��K	z8�|�1�A�:reg_large_0314/model.ckpt׈{+       ��K	��#�1�A�:reg_large_0314/model.ckptyO7z&       sO� 	PK��1�A�*

global_step/sec8?=Pe��6      "��	�EK��1�A�*�m
0
)dnn/hiddenlayer_0_fraction_of_zero_valuesq��=
�
dnn/hiddenlayer_0_activation*�   `k0@   ���A!���I�R�A)�1�S�A2�        �-���q=�z��6>u 5�9>�`�}6D>��Ő�;F>��8"uH>6��>?�J>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@�������:�           ���A              @              @              @               @      @               @      @      (@              @      @       @      @              @      (@      @               @      @               @      0@       @      @      @      8@      @      4@      0@      @      8@      @      8@      0@      4@      0@      D@      4@      H@      F@     �R@      B@      Q@      Q@      S@      X@      Y@      Z@     �a@      d@      ]@      \@      k@      `@     �m@      h@      r@     �r@     �m@     �u@     �y@     @�@     @x@     �~@     ��@     ��@      �@     `�@     ��@     ��@     `�@     ��@     ��@     Д@     P�@     `�@     `�@     Л@     ��@     �@     h�@     (�@     ��@      �@     ��@     <�@     ��@     �@     ��@     �@     �@     ��@     (�@      �@     @�@     D�@      �@     ��@     K�@     ��@     ��@    �f�@     ��@     ��@    ���@     ��@     1�@     ��@    @^�@    ���@     �@    �K�@    ��@    ���@     :�@    ���@    �}�@    �z�@    �]�@    �m�@     A�@    �L�@    �@ A    P�A    @�A    0A    �A     o
A    ��A    @�A    x�A    KA     HA    8qA    H�A    �RA    P
A    |!A    p�"A    �$A    ̠&A    ��(A    �w+A    �.A    ��0A    �E2A    (4A    �
6A    �O8A    H�:A    P=A    �@A    УAA    lzCA    �TEA    ��GA    ��IA    �\LA    ;OA   ��QA    i�RA   ��gTA    QVVA    udXA   ���ZA   ���\A    �O_A    ��`A   �	BbA   @��cA    G�dA    �<fA   � mgA   @hyhA   ��EiA    ��iA   ���iA   �3�iA   @q�hA   ���gA    l�eA   ��icA   �+�`A    ��[A    <VA    ʙPA    �GA    *�?A    F�3A    p='A    0�A    �?
A    �4�@     E�@     ��@     ��@     H�@     ��@     �f@      (@        
0
)dnn/hiddenlayer_1_fraction_of_zero_valuesq��=
�
dnn/hiddenlayer_1_activation*�   `�2@   ���A!Q:VUI��A)����U��A2�        �-���q=/�p`B>�`�}6D>��Ő�;F>��8"uH>d�V�_>w&���qa>�����0c>ڿ�ɓ�i>=�.^ol>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�������:�            J��A              @              @              @       @              0@              @      @      @       @      (@       @              @              @      @      0@      8@      0@      8@      <@       @      (@      @      8@      <@      @@      @@      4@      0@      J@      @@      V@      H@      @@      R@      Q@      V@      S@      X@     �`@      \@      a@      `@      W@     �f@     �h@      e@      j@     �q@     �r@      p@     @p@     @x@     �w@     ��@     �{@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     �@     ��@     @�@      �@     �@     �@     0�@     H�@     �@     ��@     ��@     X�@     (�@     �@     `�@     H�@     ��@     X�@     �@     Ƹ@     8�@     T�@     N�@     (�@     !�@     ��@     �@     ��@     ��@     1�@     ��@    ���@     ��@     k�@     Y�@     ��@     ��@    �k�@    ���@    @��@     �@    ���@     �@    @r�@     ��@    ���@    `/�@    �t�@    ���@    ��@    ���@    `o A    �!A    �A    �A     �A     �
A    �WA    HA    ��A    `EA    pmA    ؓA    ��A    �HA    �8A    �'!A    $�"A    $�$A    ��&A    �)A    ��+A    X�.A    P�0A    �\2A    �=4A    �Q6A    D~8A    ��:A    �=A    �?@A    2�AA    5�CA    ��EA    .�GA    0�IA    �LA    <IOA   ��,QA    a�RA   �?�TA   ��sVA   ���XA    <�ZA    a]A   ��_A   �FaA   @�dbA   ���cA   ��eA   ��bfA    ͔gA   @<�hA    �aiA   �I�iA   ���iA   @T�iA    "�hA   ��]gA   @�heA    �bA   @.+`A    �mZA    -�TA    "5NA    0�DA    ��:A    d0A    J!A     �A    �2�@    ���@     ��@     Բ@     ��@     �r@      N@        
0
)dnn/hiddenlayer_2_fraction_of_zero_values�/=>
�
dnn/hiddenlayer_2_activation*�    ��@    ���A!#�?�M*�A)�E�BE�rA2�        �-���q=�`�}6D>��Ő�;F>6��>?�J>������M>Fixі�W>4�j�6Z>cR�k�e>:�AC)8g>w`f���n>ہkVl�p>�H5�8�t>�i����v>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�           �^F�A              @              @              @              @              @               @               @      4@       @       @      @      @       @       @      0@      0@      <@      (@       @      0@      4@      L@              <@      D@      @@      D@      4@      Q@      4@      J@      F@      P@      N@      L@      B@      X@      X@      V@      R@     �a@      Z@      c@      d@      f@     �a@     �h@     �o@     @p@      m@      u@      s@     �w@     �y@     �y@     �z@     @@     `�@     ��@     ��@     ��@     ��@     `�@     P�@     0�@     ��@      �@     З@     ��@     @�@     p�@      �@     ��@     Ĥ@     h�@     ا@     �@     ��@     ,�@     ��@     0�@     T�@     t�@     d�@     T�@     ��@     ��@     ��@     ��@     X�@     ��@     R�@     ��@     �@    ���@     ��@     @�@     �@     ��@    �b�@    �|�@     ��@     ��@    ���@    ���@     ��@    ���@    �3�@     ��@     x�@    @��@    �I�@     ��@    �g�@    ���@    `EA    ��A    ��A    ��A    `�A    0bA     �A    |A    `.A    �=A    0A     2A    ��A    ؅A    8 A    T�!A    |�#A    T�%A    l�'A    �*A    ��,A    ��/A    
`1A    �3A    �5A    �7A    W9A    ��;A    
�>A    8�@A    �qBA    �ODA    �<FA    WHA    ��JA    MA    ��OA    �SQA   ���RA   �3oTA    nVA    d�WA   �DtYA    �[A    |�\A    ��]A   �2�^A    �:_A    �,_A   �gd^A   ��\A    Z�ZA   �H�WA   ��gTA   ���PA    2RJA    ��CA    (`;A    Z�1A    ��%A    ��A    PA
A    `��@    �s�@     ��@     \�@     ��@     @@      U@       @        
0
)dnn/hiddenlayer_3_fraction_of_zero_valuesg�>
�
dnn/hiddenlayer_3_activation*�   �mZ@    ��A!�q�ɥ�yA)��)��bA2�        �-���q=4��evk'>���<�)>6NK��2>�so쩾4>H��'ϱS>��x��U>d�V�_>w&���qa>BvŐ�r>�H5�8�t>�i����v>f^��`{>�����~>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�������:�            ˼�A              @               @              @              @              @      @              @              @      @      @      (@               @      (@      (@              0@      0@      0@              (@      (@       @       @              @@      D@      @@      @@      8@      @@      B@      F@      D@      J@      D@      J@      U@      W@      P@      X@      Y@      \@      U@      ^@      d@      i@     �l@     �g@      p@     �i@      p@     �p@     @v@     �t@     �x@     �}@      z@     �{@     ��@     ��@      �@     ��@     `�@      �@      �@     ��@     0�@     ��@     ��@     P�@      �@     ��@     Р@     ��@      �@     p�@     h�@     8�@     0�@     �@     ��@     p�@     ĵ@     ܸ@     $�@     L�@     h�@     ��@     ��@     ��@     �@     ��@     R�@     D�@     ��@     ��@    ���@     ��@     ��@     ��@     =�@    �!�@    ���@    ���@     ��@    ���@    ���@    ���@    @��@    �v�@    `�@    ��@    ���@    ���@    `��@     ��@    p� A    P�A    ��A    �^A    @�A    0A     2A    8kA    X(A    ��A    ��A    �-A    �|A     A    ��A    8�!A    8D#A    @4%A    ,D'A    $z)A    ,A    ��.A    ��0A    �2A    P^4A    [6A    ��8A    �:A    ��=A    nB@A    ��AA    <�CA    �EA    ��GA    7-JA    �LA    ��OA    SeQA    ��RA    xTA    �UA    ��VA   ��WA    w�VA    ��TA   ��zRA    PoNA    �GA    �@A    �)4A    'A    x�A     �A    ��@     ��@     &�@     ��@     `�@      i@      Y@      F@      D@       @        
)
"dnn/logits_fraction_of_zero_values    
�
dnn/logits_activation*�	   �Y�?   `F.@    @G\A! ܝ�+VZA)�R�9��YA2���]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�               @      @              @      @      @      4@      8@      @@      @@      T@      i@     �r@     ��@     0�@     x�@     д@     ��@     ��@    ���@    �|�@    �kA    �$A    �A     "A    ��)A    \�0A    V�3A    �M2A    I)A    �cA    @GA    �.�@     ��@     ��@     �a@      P@      <@        

loss��b?�^M+       ��K	�i��1�A�:reg_large_0314/model.ckptP�?�+       ��K	T�K�1�A�:reg_large_0314/model.ckpt<��F+       ��K	����1�A�:reg_large_0314/model.ckptyL�g+       ��K	�L�w�1�A�:reg_large_0314/model.ckpt=N��&       sO� 	�0��1�A�*

global_step/sec�^>=gU:7       ���	�81��1�A�*�m
0
)dnn/hiddenlayer_0_fraction_of_zero_valuesq��=
�
dnn/hiddenlayer_0_activation*�   �P>@   ���A!��;Df�A)W�g�I�A2�        �-���q=7'_��+/>_"s�$1>�z��6>u 5�9>������M>28���FP>��x��U>Fixі�W>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�i����v>E'�/��x>f^��`{>�����~>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@�������:�           ܱ{�A              @              @              @              @              @              @              @       @      @              @               @      @              8@      <@      4@              @@      <@      <@      0@      8@       @      (@      D@      D@      Q@      D@      4@      R@      H@      @@      F@      U@      W@      S@      N@      U@      Y@      _@      ^@     �`@      Z@     �g@     �a@     �n@      o@      o@     �v@     @s@     �t@      t@     �w@     �{@     �y@     @�@     @�@     ��@      �@     ��@     ��@     @�@     @�@     �@     �@      �@     З@     ��@     ��@     ��@     �@      �@     0�@     �@     H�@     ��@     �@     ��@     �@     ��@     x�@     ��@     �@     x�@     ��@     ��@     ��@     
�@     4�@     R�@     ��@     �@     i�@      �@     G�@     0�@     ��@     �@     B�@     ��@     �@    �&�@    @��@     ��@    �y�@    �4�@     ��@     v�@    @U�@    �R�@    `y�@    ��@    �z�@    P A    @�A    p�A     �A    `�A    �{
A    ��A    ��A    (xA    �OA    �A    �RA    �A     LA    0�A    0!A    ��"A    �$A    �&A    �(A    ,j+A    .A    ��0A    �*2A    j
4A    �6A    :>8A    �:A    <3=A    +@A    ��AA    ,mCA    �`EA    
�GA    �IA    �TLA    POA   ���PA    5�RA   ��_TA    )RVA   ��[XA   �M�ZA    p�\A   �JP_A   ���`A   �BAbA   ���cA   ���dA    �AfA    �tgA   ��~hA   @kUiA   ���iA   ��jA   @��iA   @��hA   �q�gA   ���eA    ��cA    ��`A    
�[A    �=VA   �o�PA    4�GA    T@A    D4A    �o'A     �A    �9
A    @#�@    ���@     (�@     ��@     ��@     P�@     �f@      0@        
0
)dnn/hiddenlayer_1_fraction_of_zero_valuesq��=
�
dnn/hiddenlayer_1_activation*�    �-@   ���A!>%s�ⳚA)�]�f��A2�        �-���q=%���>��-�z�!>Fixі�W>4�j�6Z>��u}��\>:�AC)8g>ڿ�ɓ�i>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�������:�           �뛱A              @              @      @              @               @      @      (@       @      @       @       @      0@       @      (@      0@               @      4@              (@       @      0@      8@      (@      B@      4@      <@      8@      4@      <@      F@      L@      F@       @      B@      N@      H@      R@      J@      W@      U@      X@      Z@      ^@      ]@     �g@     �h@     �d@      e@      i@     �n@     �o@      r@      s@     `x@     ��@      ~@     `�@     @~@     ��@     ��@     @�@     ��@     ��@     ��@      �@     А@      �@     ��@     p�@     �@     ��@     ��@     �@     ��@     �@     �@     ��@     ��@     ��@     ��@     �@     ̳@     ��@     ~�@     �@     .�@     X�@     ��@     ��@     �@     
�@     l�@     a�@     ��@     ��@     Q�@     �@    ���@     ��@     d�@     F�@    �j�@    ��@    �j�@     Z�@     ��@     ��@    �,�@     ��@     ��@     ��@    ���@    `��@    �[�@     �@    �m A    p�A    ��A     �A    �A    �Q
A    �$A    �A    @�A    ��A    �fA    H�A    P�A    �ZA    �vA    �S!A    �"A    0�$A    ��&A    �M)A    (�+A    p_.A    ��0A    fQ2A    �O4A    �'6A    D�8A    ��:A    ֺ=A    �R@A    )�AA    u�CA    2�EA    @�GA    JA    V�LA    �eOA   �'5QA   �Z�RA   ���TA   �y�VA   ��XA    �ZA    �%]A    ��_A    �aA   @9obA    ;�cA   ��)eA   @{ufA   @-�gA   @ѥhA    �uiA     �iA   @�jA   �{�iA   @��hA   ��qgA   @aseA    cA   @�3`A    |ZA    ��TA    %2NA    �DA    ��:A    ��/A    d%!A    �{A    ���@    ���@     ��@     ��@     �@      k@      T@        
0
)dnn/hiddenlayer_2_fraction_of_zero_values�/=>
�
dnn/hiddenlayer_2_activation*�   ���@    ���A!�s1R��A)bܳ���qA2�        �-���q=Z�TA[�>�#���j>������M>28���FP>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�           �ns�A              @               @              @               @      @              @              @      @      @      @              @       @       @      (@       @       @      @       @      0@       @      (@       @      @      (@      @      4@      0@      4@      B@      8@      0@      8@      F@      B@      <@      _@      T@      B@      @@      Q@      U@      U@      U@      ]@      J@      ]@      _@      a@     �`@      b@     @q@     �o@      p@      o@      p@     �r@     @x@     �~@     �y@      x@     �~@      @     �@     ��@      �@     ��@     ��@     ��@     ��@     ��@     P�@     ��@     ��@     �@     ��@     P�@     �@     ��@     �@     �@     �@      �@     ��@     ��@     ��@     X�@     l�@     4�@     ��@     ��@     <�@     �@     v�@     �@     ��@     ��@     ��@     4�@     P�@     ��@     ��@     ��@     �@     ��@    �=�@    �e�@     w�@    ���@    @��@    �R�@     �@     s�@    @�@     ��@    `��@    �k�@    ���@    �P�@    ���@    @��@     JA     A    ��A    �?A     [	A     A    p�A    �A    ��A    @�A    �}A    ��A    �ZA    ��A    � A    ��!A    4�#A    ��%A    p$(A    ��*A    �:-A    0A    <�1A    �T3A    Vd5A    �}7A    n�9A    C<A    	?A    SAA    s�BA    �DA     �FA    {�HA    r�JA    JgMA    �PA    ixQA   �w�RA   ���TA    x&VA    I�WA    #uYA    ��ZA    cN\A    �]A   �ng^A   ���^A    *�^A    \�]A    -0\A   �]�YA   �{WA   �~�SA   �3PA    O#IA    %�BA    x�9A    ��0A    �l$A    X�A    ` A     U�@     ��@     �@      �@     p�@     @w@      Q@       @        
0
)dnn/hiddenlayer_3_fraction_of_zero_valuesg�>
�
dnn/hiddenlayer_3_activation*�   ��@    ��A!C����,yA)>���49bA2�        �-���q=_"s�$1>6NK��2>������M>28���FP>��x��U>Fixі�W>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>BvŐ�r>�H5�8�t>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�������:�           8���A              @               @              @              @              @      @              @              @               @              @              @      @      @       @               @      D@      @      0@      @       @       @      (@      (@      (@      0@      8@      8@      @@      L@      D@      8@      B@      <@      8@      D@      F@      X@      H@      Q@      Q@      H@      Q@      ^@      \@      _@      a@     �c@      a@      h@      i@     �h@     �i@      q@     �r@      v@     �{@      v@     �y@     �|@      �@      �@     ��@     `�@     `�@     `�@      �@      �@     ��@      �@      �@     �@     p�@     0�@     ��@      �@     ��@     8�@     ��@     X�@     P�@     Э@     ��@      �@     ��@     x�@     �@     ��@     ��@     ^�@     ,�@     ��@     ��@     r�@     ��@     �@     ��@     z�@     ��@     2�@    ��@     [�@     ��@     ��@    ���@    ���@    ���@     ��@    ��@    ���@     ��@    ���@    ��@    ���@     ,�@     �@    @��@    ���@    ` A     �A    EA     |A     KA    Ъ	A    �dA    0YA    �+A    ��A    H�A     �A    �A    @�A    ȅA    � A    �I"A    �$A    �&A    T_(A    ��*A    08-A    �0A    �1A    �^3A    R$5A    �;7A    �9A    ~�;A    x�>A    ��@A    �mBA    &DA    /'FA    `;HA    ��JA    �1MA    �	PA    �mQA    ��RA    �WTA   �R�UA    IdVA    o�VA   ��UA   �^bTA    {�QA    �8MA    )FA    �c>A    � 3A    �%A     �A    ��A    �[�@     n�@      �@     X�@     ��@     �f@      ^@      H@      D@      4@        
)
"dnn/logits_fraction_of_zero_values    
�
dnn/logits_activation*�	    ���?    &�@    @G\A!��M;2ZA)����w�YA2�I���?����iH�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�               @              @              (@      0@      F@      @@      H@      F@      e@     �f@     @z@     @�@     ��@      �@     t�@     ��@     T�@     ��@    ���@    ���@     �A      A    x�A    �"A    �()A    �!0A    V2A    &B1A    T�(A    ��A    ��A    �,�@     ��@     ��@     �o@      W@      D@      @@        

loss�<a?�ܴ�+       ��K	/6�1�A�:reg_large_0314/model.ckpt�W�+       ��K	|)-��1�A�:reg_large_0314/model.ckptJ �+       ��K	0-A�1�A�:reg_large_0314/model.ckpt�OL&       sO� 	j�ü1�A�*

global_step/sec��>=%�Y��6      �v	&�ü1�A�*�m
0
)dnn/hiddenlayer_0_fraction_of_zero_valuesq��=
�
dnn/hiddenlayer_0_activation*�   ��F@   ���A!�����y�A)p?/&Wv�A2�        �-���q=7'_��+/>_"s�$1>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@�������:�           Bs�A               @              (@              @               @              (@               @      0@      @      @               @      (@       @      @      4@      (@      B@      0@      8@      (@      8@      B@      <@      4@      8@      J@      D@      L@      @@      J@      J@      S@      J@      ]@      [@      U@      T@      R@      W@      f@      Z@     �`@      k@      g@      e@      p@     �j@     �q@     �t@      w@      @     �z@     @y@      }@     @~@     ��@     ��@     ��@     @�@     ��@     �@     ��@     ��@     �@      �@     p�@     ��@     ��@     `�@     h�@     ��@     0�@     �@     ��@     T�@     (�@     D�@     ��@     ��@     ��@     0�@     �@     ̿@     �@     :�@     8�@     ��@     �@     ��@     P�@     ��@     ��@     �@     '�@    ���@     ��@    �[�@     B�@    �+�@     p�@    ���@     ��@     ��@    ���@    �$�@    ���@     _�@     �@    @-�@     ��@    ���@    ���@    �d A    @�A    �A    ��A    @�A    �*
A     �A    0�A    �]A    �A    �A    �A    �|A    x2A    �A     !A    ��"A    |�$A    �&A    �(A    dM+A    �6.A    ��0A    �32A    t�3A    >6A    �28A    ��:A    �U=A    �@A    ��AA    �yCA    ,QEA    GA    ��IA    �GLA    ��NA    ��PA    ^�RA    PeTA    =BVA   �_XA    ��ZA   ��\A   � U_A    7�`A   ��@bA    ��cA    ��dA   �LAfA   �Z~gA   ���hA    �_iA    ��iA   ��$jA   ���iA   @�iA    ��gA   ���eA   �;�cA   @��`A    �2\A   �DjVA    ��PA    HA    �0@A    I4A     �'A    �A    �`
A    �
�@    ���@     ��@     �@     P�@     0�@     �i@      (@        
0
)dnn/hiddenlayer_1_fraction_of_zero_valuesq��=
�
dnn/hiddenlayer_1_activation*�   `�@   ���A!������A)��Ȟ��A2�        �-���q=%���>��-�z�!>u 5�9>p
T~�;>�
L�v�Q>H��'ϱS>��x��U>��u}��\>d�V�_>w&���qa>:�AC)8g>ڿ�ɓ�i>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@�������:�           "8��A               @              @              @      @              @       @               @               @              @       @               @       @      @      @      4@       @              0@       @      8@      4@      (@      @       @      4@       @      L@      <@      @@      @@      B@      J@      D@      Q@      B@      J@      S@      S@      P@      W@     �c@      X@      T@      p@      Z@     �a@     �d@     �j@      l@     �o@     �f@      q@     �s@     �t@     �y@      v@     ��@      {@     @@     ��@      �@      �@     ��@     ��@     ��@     0�@     ��@     ��@      �@     P�@     М@     ��@     ��@     ��@     ��@     ��@     ��@     Ъ@     ��@     6�@     ��@     ��@     �@     �@     |�@     ��@     H�@     &�@     x�@     T�@     R�@     j�@     ��@     ��@     ��@     j�@     ��@     ��@     [�@     K�@    �A�@     �@    ���@     ��@    ���@    � �@     n�@    ��@    ���@    ���@    `��@    @��@    ���@    @G�@    ���@     �@    @� A    0>A    �A    P(A    @(A    �
A    `6A    �A    `�A     �A    ЀA    ЏA    H�A    �A    x~A    �I!A    �#A    ��$A    ��&A    7)A    ��+A    ��.A    ��0A    ``2A    2O4A    .\6A    ^�8A    �;A    F�=A    qQ@A    ��AA    �CA    k�EA    �GA    �JA    W�LA    kOA   �fCQA   �H�RA   �5�TA   � �VA   ���XA   ���ZA   ��4]A   �Ų_A   ��#aA   @ybA    ��cA   ��/eA   �.�fA    �gA    {�hA   ��iA   ��iA   �ajA    i�iA   @s�hA   �zgA   @�eA   @lcA   �9`A    �ZA   �I�TA    �!NA    ��DA    ��:A    �/A    ,� A    �'A    �!�@     <�@     ��@     h�@     ��@      d@      T@       @        
0
)dnn/hiddenlayer_2_fraction_of_zero_values�/=>
�
dnn/hiddenlayer_2_activation*�   @�@    ���A!����=�A)I��V{�qA2�        �-���q=����W_>>p��Dp�@>/�p`B>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>:�AC)8g>ڿ�ɓ�i>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�           �ў�A              @      @              @              @              @               @              @       @      @      (@       @       @               @              @              0@       @              (@      @       @      (@      (@      0@      8@       @      @      @@      4@      @@      H@      @@      J@      F@      L@      H@      P@      W@      J@      [@      X@      Q@      \@      `@      ]@      W@     �k@      c@      o@     �k@      p@     @p@      o@      u@     @w@      w@     @{@     �@     @@     ��@     ��@     @�@     ��@     ��@     ��@     ��@     ��@     0�@     ��@     Ж@     ��@     ��@     p�@     �@     ��@      �@     P�@     ت@     ��@     Э@     ��@     X�@     �@     ĵ@     ��@     �@     ��@     ��@     ��@     N�@     
�@     ��@     F�@     ��@     n�@     "�@    ���@     �@    �2�@     �@    ���@     P�@     R�@     ��@    �@�@     f�@    ���@    ���@     ��@    @`�@    @)�@    ���@    @��@    @�@     j�@     ��@    `��@    @oA     FA    �*A     %A    ��	A    09A    �A    �A    �A    ��A    h�A    �A    8�A    _A    � A    `h"A    �($A    L3&A    D�(A    ��*A    8�-A    �/0A    ��1A    Φ3A    Ć5A    ��7A    8�9A    ��<A    �]?A    �-AA    ��BA    ®DA    r�FA    l�HA    D	KA    ��MA   �6PA    �|QA   ��RA   �ȈTA   �*VA    ��WA   ��MYA    W�ZA    �\A    U7]A    &�]A   ��U^A    �^A   �%/]A   �8�[A   �}PYA   ��pVA    WSA    G8OA    �bHA    �AA    ��8A    X>0A    ��#A     �A    @�A    �&�@     Z�@     ��@     ��@     P�@     @r@      Q@       @        
0
)dnn/hiddenlayer_3_fraction_of_zero_valuesg�>
�
dnn/hiddenlayer_3_activation*�   ���@    ��A!3��F�xA)Ś���aA2�        �-���q=����W_>>p��Dp�@>/�p`B>w`f���n>ہkVl�p>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�������:�           �h��A              @      @               @              @      @      (@      @       @       @      @      @      @              @      @      @      @       @      2@      0@      4@      (@      (@      @      4@      0@      8@      0@      8@      4@      0@       @      D@      B@      <@      D@      B@      J@      B@      P@      N@      W@      S@      U@      \@      a@     �a@     �a@     �f@      e@     �d@     �j@     �k@      m@     @r@     �r@      r@     �v@     �}@     �z@     �@     ��@     ��@     `�@     ��@     @�@     ��@     `�@     ��@     ��@     �@     `�@     Е@      �@     ��@     �@     �@     h�@     �@     Ч@     ��@     ��@     ��@     �@     `�@     ��@     ��@     ��@     ̺@     �@     ��@     @�@     ��@     P�@     ��@     w�@     �@     ��@     >�@     %�@     �@     V�@     ��@    ��@     ��@    �&�@    �)�@    @��@    �,�@     ��@    ���@    ���@    ���@    @��@    `g�@    `��@     ��@    �U�@    ���@    pe A    �lA    `#A     �A    �A    ��
A    0eA    8A    ��A    �uA    �cA    @�A    @�A    hjA    �"A    L !A    ��"A    �$A    T�&A    �(A    `p+A    	.A    ��0A    �2A    ��3A    ��5A    ��7A    B8:A     �<A    ny?A    A+AA    l�BA    ;�DA    ��FA    ��HA    ��JA    T�MA   ��PA    |QA    U�RA    9;TA   ��aUA    �VA    4VA   �b�UA    ��SA    �jQA    -kLA    �WEA    T`=A    �O2A    ��$A    �A    �EA    ��@     ��@     ��@     �@     @�@      g@     �`@      J@      F@      B@        
)
"dnn/logits_fraction_of_zero_values    
�
dnn/logits_activation*�	    �	�?    ��	@    @G\A!`�tV*ZA)f�ճ�YA2��g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�������:�              @      @       @      (@       @      8@      <@      L@      8@      <@      T@      T@      d@      p@      }@     @�@     �@     ��@     ��@     ��@     ��@     ��@    �s�@    �c�@    �q�@    ��A    @pA    ��A    @�"A    p�(A    ��.A    �F1A    �i0A    D�(A    �uA     �A    ��@     ��@     ��@     �~@      [@      R@      H@      (@      @        

loss��_?�࿒+       ��K	���ؼ1�A�:reg_large_0314/model.ckpt�\�+       ��K	3�kr�1�A�:reg_large_0314/model.ckpt8�E?+       ��K	"���1�A�:reg_large_0314/model.ckptbj��+       ��K	HJ��1�A�:reg_large_0314/model.ckpt�ց�&       sO� 	���پ1�A�*

global_step/secf�?=.m�2g8      �^�	�پ1�A�*�p
0
)dnn/hiddenlayer_0_fraction_of_zero_valuesq��=
�
dnn/hiddenlayer_0_activation*�   �lH@   ���A!D���A)�~����A2�        �-���q=H��'ϱS>��x��U>d�V�_>w&���qa>�����0c>ڿ�ɓ�i>=�.^ol>w`f���n>BvŐ�r>�H5�8�t>�i����v>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@�������:�           �Kk�A               @              @      @              @      @              @@      @              @              @      (@      @      (@      0@      @      @              @      @       @      4@      4@      4@       @      4@      <@      8@      D@      <@      <@      @@      H@      <@      L@      R@      Q@      R@      W@      N@      Z@      X@      [@     �`@      `@      `@     �c@     �g@     �d@      j@     �i@     �h@     �o@      r@     �t@     @{@      |@      ~@      �@     ��@     ��@     `�@     `�@     ��@      �@     ��@     `�@     �@     p�@     ��@     ��@     ��@     ��@     О@     �@     P�@     �@     Ц@     0�@     Ы@     ��@     @�@     P�@     ��@     4�@     l�@     ��@     $�@     8�@     ��@     :�@     ��@     c�@     ��@     |�@     r�@     ��@     ��@    ��@     �@     ��@    ���@     O�@    �D�@    ���@     ��@     �@     ��@     l�@     d�@    ``�@    @O�@     �@    �^�@     ��@    ���@    @S�@     U A    �A    p�A    ��A    P�A    ��	A    ��A    �A    HbA    X/A    "A     -A    ��A    x0A    X�A    D� A    @�"A    �$A    ��&A    ��(A    x$+A    .A    �0A    �.2A    R4A    �5A    �8A    �:A    �'=A    @A    ��AA    �TCA    jQEA    vGA    ��IA    44LA    ��NA    )�PA   �@�RA   ��WTA    �:VA    �SXA    9~ZA   ���\A   �U_A   ���`A   �n?bA   �<�cA   @��dA   �{CfA    �zgA   �<�hA   �eoiA    ��iA    �3jA   ���iA   ��*iA   @��gA   @3fA   ��cA   �"aA    sg\A   ��VA   �-QA    =aHA    �]@A    ��4A    0�'A    �A    �r
A     �@     x�@     ��@     ��@     (�@     ��@     �i@      (@        
0
)dnn/hiddenlayer_1_fraction_of_zero_valuesq��=
�
dnn/hiddenlayer_1_activation*�   `��@   ���A!�zA&8��A)�J� ��A2�        �-���q=u 5�9>p
T~�;>�
L�v�Q>H��'ϱS>d�V�_>w&���qa>cR�k�e>:�AC)8g>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@�������:�           ���A              @              @              @              @               @      @      @      @              @       @      @      0@       @      @              @      @      (@       @              8@      0@      4@      <@      4@      4@      (@      H@      @@      D@      @@      P@      T@      J@      <@      H@      S@      [@      T@      R@      \@      \@      V@      a@      `@      ]@      b@      i@     �h@      g@     �n@     @q@      m@      y@     �s@      {@     �|@     �{@     ��@     ��@      �@     ��@     `�@      �@     �@     ��@     �@     p�@     Е@     0�@      �@     ��@     p�@     �@     �@     �@     X�@     (�@     �@     `�@     ذ@     �@     ��@     ȶ@     ��@     ��@     ��@     ^�@     ��@     L�@     ��@     8�@     ��@     g�@     �@     ��@     ��@     t�@     T�@    �B�@     ��@     ��@    �k�@    ��@     +�@     �@     V�@    @�@    ���@    ���@    `��@    ���@    ���@     ��@    ���@    ���@    Pp A    `A    @A    pA    p!A     x
A    �pA     �A    0�A    @�A    ��A    hA    `�A    ��A    �tA    �H!A    t#A    �%A    �&A    P>)A    4�+A    ��.A    r�0A    4l2A    �S4A    >a6A    ��8A    z;A    ��=A    EY@A    �AA    ��CA    �EA    �GA    �AJA    ߢLA    HpOA   �98QA   �@�RA   ���TA   ��VA    !�XA    ��ZA    |2]A   ���_A    �,aA   �
~bA   ���cA   �b:eA   ��fA   @K�gA    ��hA   �a�iA   �DjA    -jA    ��iA   �`�hA   @{gA   �{�eA    ucA   �i@`A   �8|ZA   �D�TA    �NA    z�DA    �:A    �\/A    �� A    ��A    ���@    ���@     ��@     �@      �@      d@      Q@      0@        
0
)dnn/hiddenlayer_2_fraction_of_zero_values�/=>
�
dnn/hiddenlayer_2_activation*�   `�{@    ���A!  ���A)�vd�5qA2�        �-���q=����W_>>p��Dp�@>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�           L�ĦA              @               @      @      @              @      @      @      (@              @       @       @      (@      (@       @       @      (@      0@      (@      @      4@      @       @      (@      B@      <@      0@      4@      F@      Q@      <@      <@      P@      L@      W@      L@      N@      Q@      U@      [@      V@      N@      T@      W@      a@      `@     �a@      d@      l@     �g@     �n@     �q@     �q@     @r@      w@     �w@     @x@     ��@     ��@      �@     �@     @�@     @�@      �@     ��@     ��@     P�@     �@     ��@     ��@     ��@     `�@     0�@     `�@     ��@     ��@      �@     X�@     ��@     `�@     (�@     Ȱ@     X�@     �@     ��@     ��@     \�@     ��@     H�@     \�@     ��@     ��@     ��@     ��@     ��@     F�@     �@    ���@     �@     ��@     ��@     ��@     ��@    ��@    �r�@     e�@    @��@     )�@    ���@    �L�@    ���@     ��@    ��@    ���@    ���@    ���@    ���@    �% A    ��A    �yA    ��A    ��A    `�	A    ��A     OA    P>A     A    ��A    (�A    �?A    �A    ��A    (� A    ȉ"A    (S$A    �a&A    ��(A    �+A    ��-A    �h0A    \�1A     �3A    ��5A    ��7A    &:A    F�<A    F�?A    �BAA    
�BA    b�DA    ��FA    ��HA    m(KA    ��MA    PA    <|QA    �RA    }TA    �
VA   ���WA    !YA   ��ZA   �2�[A   ��\A    �]A    M�]A    ��]A   ���\A   ��%[A   ���XA   ���UA   ���RA    ��NA    ��GA    %�AA    l[8A    ��/A    #A    88A    ��A    `��@     ��@     ��@     �@     �@     �q@      Q@       @        
0
)dnn/hiddenlayer_3_fraction_of_zero_valuesg�>
�
dnn/hiddenlayer_3_activation*�   �9�@    ��A!k/0w�xA)����aA2�        �-���q=p��Dp�@>/�p`B>������M>28���FP>H��'ϱS>��x��U>d�V�_>w&���qa>ڿ�ɓ�i>=�.^ol>�i����v>E'�/��x>f^��`{>�����~>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�           p㻒A              @              @               @              @              @              @      @      (@               @               @      (@              @@              @      @      4@      (@      4@              4@      0@      @@      4@       @      4@      H@      0@      8@       @      B@      L@      J@      R@      J@      U@      N@      W@      V@      X@      b@      Z@     �a@      b@      e@      g@     �h@      f@     �g@      h@     �r@      s@      w@     �v@      s@      y@     �@     @z@     ��@     ��@     ��@     @�@     ��@     ��@     ��@     @�@     ��@     @�@     ��@     �@     