import 'package:flutter/material.dart';

void main() {
  runApp(const SplitstarConsoleApp());
}

class SplitstarConsoleApp extends StatelessWidget {
  const SplitstarConsoleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: Scaffold(
        body: Center(child: Text('Splitstar Operations Console Dashboard')),
      ),
    );
  }
}
