import 'package:flutter/material.dart';

void main() {
  runApp(const SplitstarOpsApp());
}

class SplitstarOpsApp extends StatelessWidget {
  const SplitstarOpsApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: Scaffold(
        body: Center(child: Text('Splitstar Operations Console Dashboard')),
      ),
    );
  }
}
