def test_constants(sess_registry):
    c_units = sess_registry.speed_of_light
    assert c_units == dict(speed_of_light=1)

    q_sys = sess_registry.constants.speed_of_light
    assert (
        q_sys.magnitude == (1 * sess_registry.speed_of_light).to_base_units().magnitude
    )
    assert q_sys.units == dict(meter=1, second=-1)

    q_imp = sess_registry.sys.imperial.speed_of_light
    assert (
        q_imp.magnitude
        == (1 * sess_registry.speed_of_light).to("yard/second").magnitude
    )
    assert q_imp.units == dict(yard=1, second=-1)
