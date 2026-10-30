#!/bin/sh
# Chooses the TLS certificate that nginx serves on port 443.
#
# The official nginx image runs every executable *.sh in /docker-entrypoint.d/ (sorted) before
# starting nginx, and aborts the start if one of them fails; the "05-" prefix makes this run first.
#
#  1. If /etc/nginx/certs/server.crt and server.key are mounted (./certs:/etc/nginx/certs:ro),
#     they are checked (valid PEM, unencrypted key that matches the certificate) and used.
#  2. Otherwise a self-signed certificate is generated in /etc/nginx/certs-generated with
#     subjectAltName DNS:localhost, IP:127.0.0.1, IP:::1 plus $EXTRA_SAN
#     (e.g. EXTRA_SAN="IP:192.168.1.10,DNS:mypc.local" for use on the LAN). It is reused across
#     restarts of the same container while the SAN list is unchanged and it is not about to expire.
#
# nginx always reads the fixed paths /etc/nginx/certs-active/server.{crt,key}, which are
# symlinks to the pair chosen here. The CERT_*_DIR overrides exist only to test this script
# outside the container.
set -eu
# The entrypoint runs this script inside a `find | sort | while read` loop: never let a command
# here consume that stdin, or the scripts after this one would be skipped.
exec </dev/null

ME=$(basename "$0")
MOUNT_DIR=${CERT_MOUNT_DIR:-/etc/nginx/certs}
GEN_DIR=${CERT_GENERATED_DIR:-/etc/nginx/certs-generated}
ACTIVE_DIR=${CERT_ACTIVE_DIR:-/etc/nginx/certs-active}
# 397 days keeps the certificate within every browser/OS validity limit, even if the user
# decides to trust it explicitly.
DAYS=397
RENEW_BEFORE_S=2592000 # regenerate when less than 30 days of validity remain
BASE_SAN="DNS:localhost,IP:127.0.0.1,IP:::1"

log() {
    if [ -z "${NGINX_ENTRYPOINT_QUIET_LOGS:-}" ]; then
        echo "$ME: $*"
    fi
}

fail() {
    echo "$ME: ERRORE: $*" >&2
    exit 1
}

activate() {
    mkdir -p "$ACTIVE_DIR"
    ln -sf "$1" "$ACTIVE_DIR/server.crt"
    ln -sf "$2" "$ACTIVE_DIR/server.key"
    [ -r "$ACTIVE_DIR/server.crt" ] && [ -r "$ACTIVE_DIR/server.key" ] ||
        fail "impossibile attivare il certificato in $ACTIVE_DIR"
}

fingerprint() {
    openssl x509 -in "$1" -noout -fingerprint -sha256 2>/dev/null | sed 's/^[^=]*=//'
}

# Checks a certificate/key pair the user mounted. The key must be unencrypted (nginx would
# otherwise need ssl_password_file) and must belong to the certificate.
check_pair() {
    crt=$1
    key=$2
    openssl x509 -in "$crt" -noout 2>/dev/null ||
        fail "$crt non è un certificato PEM valido"
    crt_pub=$(openssl x509 -in "$crt" -noout -pubkey 2>/dev/null) ||
        fail "impossibile leggere la chiave pubblica di $crt"
    key_pub=$(openssl pkey -in "$key" -passin pass: -pubout 2>/dev/null) ||
        fail "$key non è una chiave privata PEM leggibile senza passphrase"
    [ "$crt_pub" = "$key_pub" ] ||
        fail "$key non corrisponde al certificato $crt"
    if ! openssl x509 -in "$crt" -noout -checkend 0 >/dev/null 2>&1; then
        log "ATTENZIONE: il certificato $crt è scaduto: il browser lo rifiuterà"
    fi
}

# Validates EXTRA_SAN and prints the normalised list (comma separated, no spaces). Only
# DNS:<name> and IP:<address> entries made of safe characters are accepted, so the value can
# never inject anything into the OpenSSL configuration below.
extra_san() {
    raw=$(printf '%s' "${EXTRA_SAN:-}" | tr -d ' \t\r\n')
    [ -n "$raw" ] || return 0
    out=""
    old_ifs=$IFS
    set -f
    IFS=,
    # shellcheck disable=SC2086 # word splitting on commas is intended
    set -- $raw
    IFS=$old_ifs
    set +f
    for entry in "$@"; do
        case "$entry" in
            "") continue ;;
            DNS:* | dns:*)
                value=${entry#*:}
                case "$value" in
                    "" | *[!A-Za-z0-9.*_-]*) fail "voce EXTRA_SAN non valida: '$entry' (nome DNS non valido)" ;;
                esac
                entry="DNS:$value"
                ;;
            IP:* | ip:*)
                value=${entry#*:}
                case "$value" in
                    "" | *[!0-9A-Fa-f:.]*) fail "voce EXTRA_SAN non valida: '$entry' (indirizzo IP non valido)" ;;
                esac
                entry="IP:$value"
                ;;
            *) fail "voce EXTRA_SAN non valida: '$entry' (usa IP:<indirizzo> o DNS:<nome>, separati da virgole)" ;;
        esac
        out="$out,$entry"
    done
    printf '%s' "${out#,}"
}

command -v openssl >/dev/null 2>&1 || fail "openssl non è installato nell'immagine"

# 1. Certificate provided by the user.
if [ -e "$MOUNT_DIR/server.crt" ] || [ -e "$MOUNT_DIR/server.key" ]; then
    [ -s "$MOUNT_DIR/server.crt" ] && [ -s "$MOUNT_DIR/server.key" ] ||
        fail "in $MOUNT_DIR servono entrambi i file server.crt e server.key (non vuoti)"
    check_pair "$MOUNT_DIR/server.crt" "$MOUNT_DIR/server.key"
    activate "$MOUNT_DIR/server.crt" "$MOUNT_DIR/server.key"
    log "uso il certificato montato in $MOUNT_DIR (SHA-256 $(fingerprint "$MOUNT_DIR/server.crt"))"
    exit 0
fi

# 2. Self-signed certificate.
extra=$(extra_san) || exit 1
san=$BASE_SAN
[ -z "$extra" ] || san="$san,$extra"

crt="$GEN_DIR/server.crt"
key="$GEN_DIR/server.key"
san_file="$GEN_DIR/san.txt"

if [ -s "$crt" ] && [ -s "$key" ] && [ -f "$san_file" ] &&
    [ "$(cat "$san_file")" = "$san" ] &&
    openssl x509 -in "$crt" -noout -checkend "$RENEW_BEFORE_S" >/dev/null 2>&1; then
    log "riuso il certificato autofirmato esistente ($san)"
else
    mkdir -p "$GEN_DIR"
    chmod 700 "$GEN_DIR"
    tmp=$(mktemp -d "$GEN_DIR/.tmp.XXXXXX")
    trap 'rm -rf "$tmp"' EXIT
    cat >"$tmp/openssl.cnf" <<EOF
[req]
prompt = no
distinguished_name = dn
x509_extensions = v3_server
default_md = sha256

[dn]
CN = localhost
O = Amplificatore in controfase (certificato autofirmato)

[v3_server]
basicConstraints = critical, CA:FALSE
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectKeyIdentifier = hash
subjectAltName = $san
EOF
    if ! err=$(umask 077 && openssl req -x509 -newkey rsa:2048 -nodes -days "$DAYS" \
        -config "$tmp/openssl.cnf" -keyout "$tmp/server.key" -out "$tmp/server.crt" 2>&1); then
        fail "generazione del certificato non riuscita: $err"
    fi
    mv -f "$tmp/server.key" "$key"
    mv -f "$tmp/server.crt" "$crt"
    printf '%s\n' "$san" >"$san_file"
    log "generato un certificato autofirmato valido $DAYS giorni per: $san"
fi

activate "$crt" "$key"
log "impronta SHA-256 del certificato: $(fingerprint "$crt")"
log "per l'uso in LAN aggiungi l'IP del computer con EXTRA_SAN (es. EXTRA_SAN=IP:192.168.1.10) oppure monta il tuo certificato in /etc/nginx/certs"
