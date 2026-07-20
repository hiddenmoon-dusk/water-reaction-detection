from water_server.security import verify_password


def test_admin_routes_require_login(client):
    response = client.get("/admin")

    assert response.status_code == 302
    assert "/admin/login" in response.headers["Location"]


def test_wrong_password_is_rejected(client):
    response = client.post("/admin/login", data={"password": "wrong"})

    assert response.status_code == 401
    assert "密码错误" in response.get_data(as_text=True)


def test_login_is_rate_limited(client):
    for _ in range(5):
        client.post("/admin/login", data={"password": "wrong"})

    response = client.post("/admin/login", data={"password": "wrong"})

    assert response.status_code == 429


def test_password_can_be_changed(admin_client, db):
    response = admin_client.post(
        "/admin/password",
        data={
            "current_password": "test-admin-password-2026",
            "new_password": "new-secure-password",
        },
    )

    assert response.status_code == 200
    state = db.execute("SELECT admin_password_hash FROM app_state WHERE id = 1").fetchone()
    assert verify_password("new-secure-password", state["admin_password_hash"])


def test_csrf_is_required_for_admin_mutations(admin_client):
    admin_client.environ_base.pop("HTTP_X_CSRF_TOKEN")

    response = admin_client.post(
        "/admin/password",
        data={
            "current_password": "test-admin-password-2026",
            "new_password": "new-secure-password",
        },
    )

    assert response.status_code == 400
