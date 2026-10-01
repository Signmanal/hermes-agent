import path from 'node:path'

import { rehashPayloadDigests } from './payload-digests.mjs'

/**
 * @param {Pick<import('app-builder-lib').ElectronSignOptions, 'entitlements' | 'entitlementsInherit' | 'hardenedRuntime'> & { ignore: (file: string) => boolean }} policy
 * @returns {(opts: import('@electron/osx-sign').SignOptions, packager: import('app-builder-lib').MacPackager) => Promise<void>}
 */
export function createMacSigner(policy) {
  return async (opts, packager) => {
    const target = opts.platform === 'mas' ? (opts.type === 'development' ? 'mas-dev' : 'mas') : 'mac'
    // app-builder-lib 27.0.0-alpha.9 added the resolved signing identity as a 4th argument; it only
    // feeds the ad-hoc (`sign.identity: "-"`) entitlement branch. buildSignOptions hands the custom
    // signer a string (`identity.hash || identity.name`), so map just that case back to an Identity.
    const identity = opts.identity === '-' ? { name: '-' } : null
    const optionsForFile = await packager.helper.getOptionsForFile(opts.app, target, policy, identity)
    const inheritedIgnore = opts.ignore
    const { sign } = await import('@electron/osx-sign')
    await sign({
      ...opts,
      ignore: file => (typeof inheritedIgnore === 'function' && inheritedIgnore(file)) || policy.ignore(file),
      // Batching defers child signatures until after all option callbacks.
      batchCodesignCalls: false,
      optionsForFile: file => {
        if (file === opts.app) {
          rehashPayloadDigests(path.join(opts.app, 'Contents', 'Resources', 'agent-payload'))
        }
        return optionsForFile(file)
      }
    })
  }
}
