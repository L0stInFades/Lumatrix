-module(benchmark_clock_ffi).
-export([monotonic_milliseconds/0]).

monotonic_milliseconds() ->
    erlang:monotonic_time(microsecond) / 1000.
