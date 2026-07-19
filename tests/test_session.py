"""Signed session cookie — a browser must not be able to forge another user's id."""

from app import session as sess


def test_roundtrip_valid_token():
    token = sess.make_token(7)
    assert sess.read_token(token) == 7


def test_rejects_missing_or_malformed():
    assert sess.read_token(None) is None
    assert sess.read_token("") is None
    assert sess.read_token("7") is None            # no signature
    assert sess.read_token("notanid.sig") is None  # non-numeric id


def test_rejects_tampered_signature():
    token = sess.make_token(7)
    forged = "9." + token.split(".", 1)[1]   # keep user 7's signature, claim to be user 9
    assert sess.read_token(forged) is None

    bad_sig = "7." + ("0" * 64)              # right shape, wrong signature
    assert sess.read_token(bad_sig) is None
