from gpush.push.limiter import * 
import pytest

@pytest.mark.parametrize("output,limiter,result", [([1,2,-3,3,4,-5],SizeLimiter(magnitude=3),[1,2,-3,3,3,-3]),
                                                   ({"one":[1,2,3,-4,5]}, SizeLimiter(high=2), {"one":[1,2,2,-4,2]}),
                                                   ({"one":[-1,2,3,4,5]}, SizeLimiter(low=3), {"one":[3,3,3,4,5]}),
                                                   ({"one":[1,2,3,4,5],"two":[-1,20]}, SizeLimiter(low=3, high=5), {"one":[3,3,3,4,5],"two":[3,5]}),
                                                   (jnp.array([[1,-10,4],[1,2,3.45]]), SizeLimiter(magnitude=3), jnp.array([[1,-3,3],[1,2,3.]])),
                                                   ([10,2,jnp.array([2,30.1,5]),jnp.array(7.3)], SizeLimiter(low=3, high=7), [7,3,jnp.array([3,7.,5]),jnp.array(7.)])])
def test_size_limiter(output, limiter: SizeLimiter, result):
    test_fn = limiter.limit(lambda:output)
    res = test_fn() 
    if isinstance(res, list) or isinstance(res,tuple):
        for r,r_ in zip(res,result):
            assert jnp.allclose(r,r_)
    if isinstance(res,dict):
        for k in res.keys():
            for r,r_ in zip(res[k],result[k]):
                assert jnp.allclose(r,r_)


@pytest.mark.parametrize("output,limiter,result", [([1,2,3,4,5],GrowthLimiter(3),[1,2,3]),
                                                   ({"one":[1,2,3,4,5]}, GrowthLimiter(2), {"one":[1,2]}),
                                                   ({"one":[1,2,3,4,5],"two":[1,2]}, GrowthLimiter(5), {"one":[1,2,3],"two":[1,2]}),
                                                   ({"one":[1,2,3,4,5],"two":[1,2]}, GrowthLimiter(3), {"one":[1],"two":[1]}),
                                                   ({"one":[1,2,3,4,5,6],"two":[1,2,3,4,5], "three":[1,2,3]}, GrowthLimiter(12), {"one":[1,2,3,4],"two":[1,2,3,4], "three":[1,2,3]}),
                                                   ({"one":[1,2,3,4,5],"two":[1,2,3,4], "three":[1,2,3]}, GrowthLimiter(7), {"one":[1,2],"two":[1,2], "three":[1,2]}),
                                                   ({"one":[1,2,3,4,5],"two":[1,2,3,4,5], "three":[1,2,3]}, GrowthLimiter(11), {"one":[1,2,3,4],"two":[1,2,3,4], "three":[1,2,3]})])
def test_growth_limiter(output, limiter: GrowthLimiter, result):
    test_fn = limiter.limit(lambda:output)
    assert test_fn()==result

@pytest.mark.parametrize("output,limiter,result", [([1,2,3,4,5],StackLimiter(3),[1,2,3,4,5]),
                                                   ([1,2,3,4,5],StackLimiter(1),[1,2,3,4,5]),
                                                   ([1,2,3,4,5],StackLimiter(0),[]),
                                                   ({"one":[1,2,3,4,5]}, StackLimiter(2), {"one":[1,2]}),
                                                   ({"one":[1,2,3,4,5],"two":[1,2]}, StackLimiter(4), {"one":[1,2,3,4],"two":[1,2]}),
                                                   ({"one":[1,2,3,4,5],"two":[1,2]}, StackLimiter(0), {"one":[],"two":[]})])
def test_stack_limiter(output, limiter: StackLimiter, result):
    test_fn = limiter.limit(lambda:output)
    assert test_fn()==result