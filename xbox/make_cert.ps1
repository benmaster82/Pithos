# Creates the self-signed certificate that signs the Xbox package (build-xbox\pithos_xbox.pfx).
# Its subject must match Publisher="CN=Pithos" in Package.appxmanifest. Developer Mode installs
# packages signed this way; the certificate is only kept in the current user's store.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root 'build-xbox'
New-Item -ItemType Directory -Force $out | Out-Null
$cert = New-SelfSignedCertificate -Type Custom -Subject 'CN=Pithos' -KeyUsage DigitalSignature `
    -FriendlyName 'Pithos Xbox package' -CertStoreLocation 'Cert:\CurrentUser\My' `
    -TextExtension @('2.5.29.37={text}1.3.6.1.5.5.7.3.3', '2.5.29.19={text}')
$pwd = ConvertTo-SecureString -String 'pithos' -Force -AsPlainText
Export-PfxCertificate -Cert $cert -FilePath (Join-Path $out 'pithos_xbox.pfx') -Password $pwd | Out-Null
Write-Host "certificate $($cert.Thumbprint) -> $out\pithos_xbox.pfx"
