-module(non_finite_test_values_ffi).
-export([nan/0, infinity/0]).

%% The BEAM rejects non-finite floats at its runtime boundary, so these tests
%% exercise the JavaScript-only values and take the finite branch on Erlang.
nan() -> 0.0.
infinity() -> 0.0.
